import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union

from fast_pytorch_kmeans import KMeans
from pkg_resources import packaging

from loss.loss import ContrastiveLoss, PixelContrastiveLoss
from models.dino import vision_transformer as vits
from models.dino.utils import load_pretrained_weights
from models.model_util import normalize_minmax
from models.open_clip import create_model
from models.open_clip.tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<start_of_text>"]
    eot_token = _tokenizer.encoder["<end_of_text>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class Net(nn.Module):
    def __init__(self, aff_classes=36, args=None):
        super(Net, self).__init__()

        self.aff_classes = aff_classes
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.alpha = args.alpha
        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

        self.vit_feat_dim = 384
        self.cluster_num = 3
        self.stride = 16
        self.patch = 16

        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0)
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch)

        self.net = create_model('ViT-B/16', pretrained='openai')

        if args.divide == "Seen":
            self.classnames = ['beat', "boxing", "brush_with", "carry", "catch",
                         "cut", "cut_with", "drag", 'drink_with', "eat",
                         "hit", "hold", "jump", "kick", "lie_on", "lift",
                         "look_out", "open", "pack", "peel", "pick_up",
                         "pour", "push", "ride", "sip", "sit_on", "stick",
                         "stir", "swing", "take_photo", "talk_on", "text_on",
                         "throw", "type_on", "wash", "write"]
        elif args.divide == "Unseen":
            self.classnames = ["carry", "catch", "cut", "cut_with", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                             "swing", "take_photo", "throw", "type_on", "wash"]
        else:  # HICO-IIF
            self.classnames = ['cut_with', 'drink_with', 'hold', 'open', 'pour', 'sip', 'stick', 'stir', 'swing', 'type_on']

        self.aff_proj = Mlp(
            in_features=self.vit_feat_dim,
            hidden_features=int(self.vit_feat_dim * 4),
            act_layer=nn.GELU,
            drop=0.,
        ).cuda()
        self.aff_classifier = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1).cuda()
        self.K_contrast_projection = nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, 1).cuda()
        self.pixel_contrast_projection = nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, 1).cuda()

        self.aff_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )

        self.CELoss = nn.CrossEntropyLoss().cuda()
        self.ContrastiveLoss = ContrastiveLoss(temperature=args.cont_temperature).cuda()
        self.PixelContrastiveLoss = PixelContrastiveLoss(temperature=args.cont_temperature).cuda()

    @staticmethod
    def _select_class_map(pred, labels):
        batch_idx = torch.arange(labels.size(0), device=pred.device)
        return pred[batch_idx, labels]

    @staticmethod
    def _local_mean(feat_map):
        out = torch.empty_like(feat_map)
        h, w = feat_map.shape[-2:]

        for i in range(h):
            start_h = max(i - 1, 0)
            end_h = i + 2
            for j in range(w):
                start_w = max(j - 1, 0)
                end_w = j + 2
                neighborhood = feat_map[..., start_h:end_h, start_w:end_w]
                out[..., i, j] = neighborhood.mean(dim=(-2, -1))

        return out

    @staticmethod
    def _copy_clip_fallback(batch_idx, num_exo, ego_prob_maps, ego_prob_maps2, prob_maps, prob_maps2, clip_ego_similarity, clip_exo_similarity):
        ego_prob_maps[batch_idx] = clip_ego_similarity[batch_idx]
        ego_prob_maps2[batch_idx] = clip_ego_similarity[batch_idx]
        for n_idx in range(num_exo):
            prob_maps[batch_idx, n_idx] = clip_exo_similarity[batch_idx, n_idx]
            prob_maps2[batch_idx, n_idx] = clip_exo_similarity[batch_idx, n_idx]

    @torch.no_grad()
    def get_clip_affinity_map_ego(self, ego_feats, aff_label_feats):
        b, hw, _ = ego_feats.size()
        h = w = int(hw ** 0.5)

        ego_feats = F.normalize(ego_feats, dim=-1)
        aff_label_feats = F.normalize(aff_label_feats, dim=-1)

        ego_similarity = torch.einsum('bnc,bc->bn', ego_feats, aff_label_feats).reshape(b, h, w)
        ego_similarity = self.normalize(ego_similarity)

        return ego_feats, ego_similarity.detach()

    @torch.no_grad()
    def get_clip_affinity_map(self, exo_feats, ego_feats, aff_label_feats, exo_text_features):
        b = ego_feats.size(0)
        b_exo, hw, _ = exo_feats.size()
        h = w = int(hw ** 0.5)
        num_exo = b_exo // b

        exo_feats = exo_feats.reshape(b, -1, exo_feats.size(-1))

        exo_feats = F.normalize(exo_feats, dim=-1)
        ego_feats = F.normalize(ego_feats, dim=-1)
        aff_label_feats = F.normalize(aff_label_feats, dim=-1)
        exo_text_features = F.normalize(exo_text_features, dim=-1)

        exo_similarity_obj = torch.einsum('bnc,bc->bn', exo_feats, exo_text_features).reshape(b, num_exo, h, w)
        exo_similarity_aff = torch.einsum('bnc,bc->bn', exo_feats, aff_label_feats).reshape(b, num_exo, h, w)

        exo_similarity_neighbor = self._local_mean(exo_similarity_obj) * exo_similarity_aff
        ego_similarity = torch.einsum('bnc,bc->bn', ego_feats, aff_label_feats).reshape(b, h, w)

        return ego_similarity.detach(), exo_similarity_neighbor.detach()

    def organize_classnames(self, classnames, aff_label, prefix=None, sub_label=None):
        classnames_array = np.array(classnames)
        aff_classnames_str = classnames_array[aff_label.cpu().numpy()]

        if sub_label is not None:
            if prefix is not None:
                aff_classnames_str = [
                    prefix + ' ' + classname + ('' if sub_label in classname else ' ' + sub_label)
                    for classname in aff_classnames_str
                ]
            else:
                aff_classnames_str = [
                    classname + ('' if sub_label in classname else ' ' + sub_label)
                    for classname in aff_classnames_str
                ]
        else:
            if prefix is not None:
                aff_classnames_str = [prefix + ' ' + classname for classname in aff_classnames_str]
            else:
                aff_classnames_str = [classname for classname in aff_classnames_str]

        aff_classnames = torch.cat([tokenize(ac) for ac in aff_classnames_str]).to(aff_label.device)
        return aff_classnames, aff_classnames_str

    def forward(self, exo, ego, aff_label, epoch):
        bs, num_exo = exo.shape[:2]
        device = ego.device
        exo = exo.flatten(0, 1)

        with torch.no_grad():
            aff_classnames, _ = self.organize_classnames(
                self.classnames,
                aff_label,
                prefix='an item to',
                sub_label='with',
            )
            text_features = self.net.encode_text(aff_classnames).to(device)

            exo_aff_classnames, _ = self.organize_classnames(
                self.classnames,
                aff_label,
                prefix='a person',
                sub_label='an item',
            )
            exo_text_features = self.net.encode_text(exo_aff_classnames).to(device)

        with torch.no_grad():
            _, exo_image_features, _ = self.net.encode_image(exo, model_type='ClearCLIP', ignore_residual=True)
            _, ego_image_features, _ = self.net.encode_image(ego, model_type='ClearCLIP', ignore_residual=True)

        egofeat = ego_image_features.detach().clone()
        exofeat = exo_image_features.detach().clone()

        CLIP_ego_similarity, CLIP_exo_similarity = self.get_clip_affinity_map(
            exo_image_features,
            ego_image_features,
            text_features,
            exo_text_features,
        )

        feat_h, feat_w = CLIP_ego_similarity.shape[-2:]
        CLIP_exo_similarity = CLIP_exo_similarity.reshape(-1, feat_h, feat_w)

        with torch.no_grad():
            _, ego_key, ego_attn = self.vit_model.get_last_key(ego)
            _, exo_key, _ = self.vit_model.get_last_key(exo)
            ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()
            exo_desc = exo_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()

        ego_proj = self.aff_proj(ego_desc[:, 1:])
        exo_proj = self.aff_proj(exo_desc[:, 1:])

        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        exo_desc = self._reshape_transform(exo_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)
        exo_proj = self._reshape_transform(exo_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.size()

        ego_cls_attn = ego_attn[:, :, 0, 1:].reshape(bs, 6, feat_h, feat_w)
        ego_cls_attn = (ego_cls_attn > ego_cls_attn.flatten(-2, -1).mean(-1, keepdim=True).unsqueeze(-1)).float()
        ego_sam = ego_cls_attn[:, [0, 1, 3]].mean(1)
        ego_sam = normalize_minmax(ego_sam)
        ego_sam_flat = ego_sam.flatten(-2, -1)
        sam_hard = (ego_sam_flat > torch.mean(ego_sam_flat, 1, keepdim=True)).float()

        CLIP_ego_similarity_norm = self.normalize(CLIP_ego_similarity)
        CLIP_ego_similarity_norm = (CLIP_ego_similarity_norm > self.gamma2).type(torch.float32)

        ego_proj_copy = ego_proj.clone()
        exo_proj_copy = exo_proj.clone()

        ego_proj = self.aff_ego_proj(ego_proj)
        exo_proj = self.aff_exo_proj(exo_proj)

        ego_proj_nocond = self.aff_ego_proj(ego_proj_copy)
        exo_proj_nocond = self.aff_exo_proj(exo_proj_copy)

        ego_pred_Kcontrast = self.K_contrast_projection(ego_proj_nocond)
        exo_pred_Kcontrast = self.K_contrast_projection(exo_proj_nocond)
        ego_pred_cont_pixel = self.pixel_contrast_projection(ego_proj_nocond)

        ego_pred = self.aff_classifier(ego_proj)
        aff_logits_ego = self.gap(ego_pred).reshape(bs, self.aff_classes)
        exo_pred = self.aff_classifier(exo_proj)
        aff_logits_exo = self.gap(exo_pred).reshape(bs, num_exo, self.aff_classes)

        loss_ce_exo = torch.tensor([0.], device=device)
        for n in range(num_exo):
            loss_ce_exo = loss_ce_exo + self.CELoss(aff_logits_exo[:, n], aff_label)
        loss_ce_exo = loss_ce_exo / 3.

        loss_ce_ego = self.CELoss(aff_logits_ego, aff_label)

        exo_aff_label = aff_label.repeat_interleave(num_exo, dim=0)
        ego_pred_gt = self._select_class_map(ego_pred, aff_label)
        exo_pred_gt = self._select_class_map(exo_pred, exo_aff_label)

        if epoch != 0:
            CLIP_ego_similarity_new = CLIP_ego_similarity.detach().clone()
            CLIP_exo_similarity_new = CLIP_exo_similarity.detach().clone()
            CLIP_ego_similarity_new[CLIP_ego_similarity < 0.] = 0.
            CLIP_exo_similarity_new[CLIP_exo_similarity < 0.] = 0.
            ego_pred_gt[ego_pred_gt < 0.] = 0.
            exo_pred_gt[exo_pred_gt < 0.] = 0.
            exo_mask_gt = CLIP_exo_similarity_new * exo_pred_gt
        else:
            exo_mask_gt = CLIP_exo_similarity

        ego_desc_flat = (ego_desc * CLIP_ego_similarity.unsqueeze(1)).flatten(-2, -1)
        exo_desc_re_flat = exo_desc.reshape(b, num_exo, c, h, w).flatten(-2, -1)
        ego_desc_re_flat = ego_desc.reshape(b, 1, c, h, w).flatten(-2, -1)

        prob_maps = torch.zeros(b, num_exo, h, w, device=device)
        ego_prob_maps = torch.zeros(b, h, w, device=device)
        prob_maps2 = torch.zeros(b, num_exo, h, w, device=device)
        ego_prob_maps2 = torch.zeros(b, h, w, device=device)
        ego_part_idx = torch.ones(b, device=device)

        for b_idx in range(b):
            exo_aff_desc = []
            for n_idx in range(num_exo):
                tmp_cam = exo_mask_gt[b_idx, n_idx].reshape(-1)
                tmp_cam = (tmp_cam - tmp_cam.min()) / (tmp_cam.max() - tmp_cam.min() + 1e-10)
                tmp_desc = exo_desc_re_flat[b_idx, n_idx]
                tmp_top_desc = tmp_desc[:, torch.where(tmp_cam > self.gamma1)[0]].T
                exo_aff_desc.append(tmp_top_desc)
            exo_aff_desc = torch.cat(exo_aff_desc, dim=0)

            if exo_aff_desc.shape[0] < self.cluster_num:
                ego_part_idx[b_idx] = 0
                self._copy_clip_fallback(
                    b_idx,
                    num_exo,
                    ego_prob_maps,
                    ego_prob_maps2,
                    prob_maps,
                    prob_maps2,
                    CLIP_ego_similarity,
                    CLIP_exo_similarity,
                )
                continue

            kmeans = KMeans(n_clusters=self.cluster_num, mode='euclidean', max_iter=300)
            kmeans.fit_predict(exo_aff_desc.contiguous())
            clu_cens = F.normalize(kmeans.centroids, dim=1)

            exo_sim_maps = []
            for n_idx in range(num_exo):
                exo_sim_maps.append(torch.mm(clu_cens, F.normalize(exo_desc_re_flat[b_idx, n_idx], dim=0)))
            exo_sim_maps = torch.stack(exo_sim_maps, dim=0)

            sim_map = torch.mm(clu_cens, F.normalize(ego_desc_flat[b_idx], dim=0))
            tmp_sim_max = torch.max(sim_map, dim=-1, keepdim=True)[0]
            tmp_sim_min = torch.min(sim_map, dim=-1, keepdim=True)[0]
            sim_map_norm = (sim_map - tmp_sim_min) / (tmp_sim_max - tmp_sim_min + 1e-12)

            sim_map_hard = (sim_map_norm > torch.mean(sim_map_norm, 1, keepdim=True)).float()
            inter = (sim_map_hard * sam_hard[b_idx]).sum(1)
            union = sim_map_hard.sum(1) + sam_hard[b_idx].sum() - inter
            p_score = (inter / (sim_map_hard.sum(1) + 1e-10) + sam_hard[b_idx].sum() / (union + 1e-10)) / 2

            if p_score.max() < self.alpha:
                ego_part_idx[b_idx] = 0
                self._copy_clip_fallback(
                    b_idx,
                    num_exo,
                    ego_prob_maps,
                    ego_prob_maps2,
                    prob_maps,
                    prob_maps2,
                    CLIP_ego_similarity,
                    CLIP_exo_similarity,
                )
                continue

            target_cluster = torch.argmax(p_score)
            prob_maps[b_idx] = exo_sim_maps[:, target_cluster, :].reshape(num_exo, feat_h, feat_w)
            ego_prob_maps[b_idx] = sim_map[target_cluster].reshape(feat_h, feat_w)

            target_centroid = clu_cens[target_cluster]
            flattened_exo_desc = exo_desc_re_flat[b_idx].permute(0, 2, 1).reshape(num_exo * h * w, -1)
            distances = torch.norm(flattened_exo_desc - target_centroid, dim=1)
            flattened_ego_desc = ego_desc_re_flat[b_idx].permute(0, 2, 1).reshape(h * w, -1)
            ego_distances = torch.norm(flattened_ego_desc - target_centroid, dim=1)

            prob_maps2[b_idx] = torch.softmax(-distances, dim=0).reshape(num_exo, h, w)
            ego_prob_maps2[b_idx] = torch.softmax(-ego_distances, dim=0).reshape(h, w)

        CLIP_exo_similarity_kmeans_prob = prob_maps.type(torch.float32).reshape(b * num_exo, h, w)
        CLIP_ego_similarity_kmeans_prob = ego_prob_maps.type(torch.float32).reshape(b, h, w)
        CLIP_exo_similarity_kmeans_prob2 = prob_maps2.type(torch.float32).reshape(b * num_exo, h, w)
        CLIP_ego_similarity_kmeans_prob2 = ego_prob_maps2.type(torch.float32).reshape(b, h, w)

        CLIP_ego_similarity_kmeans_prob = (CLIP_ego_similarity_kmeans_prob + CLIP_ego_similarity_kmeans_prob2) / 2
        CLIP_exo_similarity_kmeans_prob = (CLIP_exo_similarity_kmeans_prob + CLIP_exo_similarity_kmeans_prob2) / 2

        if epoch != 0:
            Kego_mask_gt = CLIP_ego_similarity_kmeans_prob * ego_pred_gt
            Kexo_mask_gt = CLIP_exo_similarity_kmeans_prob * exo_pred_gt
        else:
            Kego_mask_gt = CLIP_ego_similarity_kmeans_prob
            Kexo_mask_gt = CLIP_exo_similarity_kmeans_prob

        loss_pixelcont, _ = self.PixelContrastiveLoss(
            ego_pred_cont_pixel,
            num_exo,
            CLIP_ego_similarity_norm,
            text_features,
            egofeat,
            exofeat,
        )

        ego_part_indices = torch.nonzero(ego_part_idx.reshape(-1)).reshape(-1)
        pred_ego_bg_similarity = torch.ones_like(Kego_mask_gt).to(device) - Kego_mask_gt
        pred_exo_bg_similarity = torch.ones_like(Kexo_mask_gt).to(device) - Kexo_mask_gt
        obj_ego_similarity = torch.ones_like(CLIP_ego_similarity).to(device)

        if len(ego_part_indices) != 0:
            obj_ego_similarity[ego_part_indices] = CLIP_ego_similarity[ego_part_indices]

        loss_protocont = self.ContrastiveLoss(
            ego_pred_Kcontrast,
            exo_pred_Kcontrast,
            CLIP_ego_similarity_kmeans_prob,
            CLIP_exo_similarity_kmeans_prob,
            pred_ego_bg_similarity,
            pred_exo_bg_similarity,
            aff_label,
            num_exo,
            obj_ego_similarity,
        )

        logits = {'aff_exo': aff_logits_exo, 'aff_ego': aff_logits_ego}
        return logits, loss_ce_ego, loss_ce_exo, loss_pixelcont, loss_protocont

    @torch.no_grad()
    def test_forward(self, ego, aff_label):
        aff_classnames, _ = self.organize_classnames(
            self.classnames,
            aff_label,
            prefix='an item to',
            sub_label='with',
        )
        text_features = self.net.encode_text(aff_classnames).to(ego.device)

        _, ego_image_features, _ = self.net.encode_image(ego, model_type='ClearCLIP', ignore_residual=True)
        _, CLIP_ego_similarity = self.get_clip_affinity_map_ego(ego_image_features, text_features)

        _, ego_key, _ = self.vit_model.get_last_key(ego)
        ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1)
        ego_proj = self.aff_proj(ego_desc[:, 1:])
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_classifier(ego_proj)

        ego_map_pred = self._select_class_map(ego_pred, aff_label)
        ego_map_pred_mean = self._local_mean(ego_map_pred)

        refined_CLIP_ego_ego = CLIP_ego_similarity * ego_map_pred
        refined_CLIP_ego_mean = CLIP_ego_similarity * ego_map_pred_mean

        return ego_map_pred, refined_CLIP_ego_ego, refined_CLIP_ego_mean

    def _reshape_transform(self, tensor, patch_size, stride):
        height = (224 - patch_size) // stride + 1
        width = (224 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result

    def normalize(self, feat_map):
        map_norm = torch.zeros_like(feat_map).to(feat_map.device)
        for i in range(len(feat_map)):
            map_norm[i] = (feat_map[i] - feat_map[i].min()) / (feat_map[i].max() - feat_map[i].min())
        return map_norm
