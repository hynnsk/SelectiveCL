import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_log_prob(anchor_dot_contrast, logits_mask):
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    exp_logits = torch.exp(logits) * logits_mask
    return logits - torch.log(exp_logits.sum(1, keepdim=True))


class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(PixelContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        contfeat_ego,
        num_exo,
        weight_map,
        text_features,
        ego_image_features,
        exo_image_features,
    ):
        device = contfeat_ego.device
        bs, dim, feat_h, feat_w = contfeat_ego.size()
        dim_clip = exo_image_features.size(-1)

        text_features = F.normalize(text_features, dim=-1)

        exo_feats = exo_image_features.reshape(bs, -1, dim_clip)
        exo_feats = F.normalize(exo_feats, dim=-1)
        exo_similarity = torch.einsum('bnc,bc->bn', exo_feats, text_features).reshape(bs, num_exo, feat_h, feat_w)
        exo_similarity = exo_similarity.reshape(bs * num_exo, feat_h * feat_w)
        exo_sim_max = exo_similarity.max(dim=1).values.reshape(bs, num_exo)
        exo_sim_max_mean = torch.min(exo_sim_max, dim=1).values.unsqueeze(1)

        ego_feats = ego_image_features.reshape(bs, -1, dim_clip)
        ego_feats = F.normalize(ego_feats, dim=-1)
        ego_similarity = torch.einsum('bnc,bc->bn', ego_feats, text_features).reshape(bs, feat_h * feat_w)

        is_fg = (exo_sim_max_mean < ego_similarity).type(torch.float)
        weight_map = weight_map.view(bs, -1)

        contfeat_ego = F.normalize(contfeat_ego, dim=1)
        contfeat_ego = contfeat_ego.permute(0, 2, 3, 1).contiguous().view(bs, -1, dim)

        logits_mask_one = torch.scatter(
            torch.ones((feat_h * feat_w, feat_h * feat_w), device=device),
            1,
            torch.arange(feat_h * feat_w, device=device).view(-1, 1),
            0,
        )

        loss_ego_all = torch.tensor(0., device=device)
        pos_num_all = torch.tensor(0., device=device)

        for i in range(bs):
            if torch.sum(is_fg[i]) == 0:
                is_fg_i = weight_map[i].view(-1, 1)
            else:
                is_fg_i = is_fg[i].view(-1, 1)

            mask_ego_one = torch.eq(is_fg_i, is_fg_i.T) * logits_mask_one
            anchor_dot_contrast = torch.div(torch.matmul(contfeat_ego[i], contfeat_ego[i].T), self.temperature)
            log_prob = _compute_log_prob(anchor_dot_contrast, logits_mask_one)

            nonzero_idx = torch.where(mask_ego_one.sum(1) != 0.)
            mean_log_prob_pos = (mask_ego_one[nonzero_idx] * log_prob[nonzero_idx]).sum(1) / mask_ego_one[nonzero_idx].sum(1)

            weight_map_i = weight_map[i][nonzero_idx]
            fg_index = weight_map_i > 0.5
            mean_log_prob_pos = mean_log_prob_pos[fg_index]

            pos_num = torch.sum(mask_ego_one) / mask_ego_one.size(0)
            pos_num_all += pos_num

            loss_ego = -(self.temperature / self.base_temperature) * mean_log_prob_pos
            loss_ego_all += loss_ego.mean()

        loss_ego_all /= bs
        pos_num_all /= bs

        return loss_ego_all, pos_num_all


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    @staticmethod
    def _build_proto(feat_map, similarity_map):
        proto = torch.mean(feat_map * similarity_map.unsqueeze(1), dim=(2, 3))
        return F.normalize(proto, dim=1)

    def forward(
        self,
        ego_pred,
        exo_pred,
        ego_similarity,
        exo_similarity,
        ego_bg_similarity,
        exo_bg_similarity,
        aff_label,
        num_exo,
        obj_ego_similarity,
    ):
        device = ego_pred.device
        bs = aff_label.size(0)

        aff_label = aff_label.contiguous().view(-1, 1)
        aff_label_exo = aff_label.repeat_interleave(num_exo, dim=0)

        if obj_ego_similarity is not None:
            ego_anchor = torch.mean(ego_pred * obj_ego_similarity.unsqueeze(1), dim=(2, 3))
            ego_anchor = F.normalize(ego_anchor, dim=1)
        else:
            ego_anchor = torch.mean(ego_pred, dim=(2, 3))
            ego_anchor = F.normalize(ego_anchor, dim=1)

        ego_proto_pos = self._build_proto(ego_pred, ego_similarity)
        ego_proto_neg = self._build_proto(ego_pred, ego_bg_similarity)
        exo_proto_pos = self._build_proto(exo_pred, exo_similarity)
        exo_proto_neg = self._build_proto(exo_pred, exo_bg_similarity)

        feat_whole = torch.cat((ego_proto_pos, ego_proto_neg, exo_proto_pos, exo_proto_neg), dim=0)

        mask_1 = torch.eq(aff_label, aff_label.T).float().to(device)
        mask_2 = torch.zeros_like(mask_1)
        mask_2_ignore = mask_1.clone().detach()
        mask_3 = torch.eq(aff_label, aff_label_exo.T).float().to(device)
        mask_4 = torch.zeros_like(mask_3)
        mask_4_ignore = mask_3.clone().detach()

        mask_whole = torch.cat((mask_1, mask_2, mask_3, mask_4), dim=1).detach()

        logits_mask = torch.ones_like(mask_whole)
        logits_mask[:, :bs] = torch.scatter(
            torch.ones_like(mask_1),
            1,
            torch.arange(bs, device=device).view(-1, 1),
            0,
        )
        mask_whole = mask_whole * logits_mask

        ignore_mask = torch.ones_like(mask_whole)
        ignore_mask[:, bs:2 * bs] = mask_2_ignore
        ignore_mask[:, 2 * bs + (num_exo * bs):] = mask_4_ignore

        neglect_mask = torch.logical_or(mask_whole, ignore_mask).type(torch.float16)
        neglect_logits_mask = neglect_mask * logits_mask

        anchor_dot_contrast = torch.div(torch.matmul(ego_anchor, feat_whole.T), self.temperature)
        log_prob = _compute_log_prob(anchor_dot_contrast, neglect_logits_mask)

        nonzero_idx = torch.where(mask_whole.sum(1) != 0.)
        mean_log_prob_pos = (mask_whole[nonzero_idx] * log_prob[nonzero_idx]).sum(1) / mask_whole[nonzero_idx].sum(1)

        loss_ego = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss_ego.mean()
