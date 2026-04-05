import os
import sys
import time
import shutil
import logging
import argparse

import cv2
import torch
import torch.nn as nn
import numpy as np
from models.locate import Net as model

from utils.util import set_seed, process_gt, normalize_map, get_optimizer
from utils.evaluation import cal_kl, cal_sim, cal_nss, AverageMeter, compute_cls_acc

# import wandb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/DATA/AGD20K')
parser.add_argument('--save_root', type=str, default='save_models')
parser.add_argument("--divide", type=str, default="Seen", choices=["Seen", "Unseen", "HICO"])
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
##  dataloader
parser.add_argument('--num_workers', type=int, default=8)
##  train
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--show_step', type=int, default=500)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--debug', action='store_true', default=False)

parser.add_argument('--exp_name', type=str, default='SCL')
parser.add_argument('--cont_temperature', type=float, default=0.5)

parser.add_argument('--gamma1', type=float, default=0.6)
parser.add_argument('--gamma2', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.6)

#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)

args = parser.parse_args()
torch.cuda.set_device('cuda:' + args.gpu)

if args.divide == "Seen":
    aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                "talk_on", "text_on", "throw", "type_on", "wash", "write"]
    args.num_classes = 36
elif args.divide == "Unseen":
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                "swing", "take_photo", "throw", "type_on", "wash"]
    args.num_classes = 25
else: # HICO-IIF
    aff_list = ['cut_with', 'drink_with', 'hold', 'open', 'pour', 'sip', 'stick', 'stir', 'swing', 'type_on']
    args.num_classes = 10
    args.data_root = '/DATA/HICO-IIF'

if args.divide == "HICO":
    args.exocentric_root = os.path.join(args.data_root, "trainset", "exocentric")
    args.egocentric_root = os.path.join(args.data_root, "trainset", "egocentric")
    args.test_root = os.path.join(args.data_root, "testset", "egocentric")
    args.mask_root = os.path.join(args.data_root, "testset", "GT")
else:
    args.exocentric_root = os.path.join(args.data_root, args.divide, "trainset", "exocentric")
    args.egocentric_root = os.path.join(args.data_root, args.divide, "trainset", "egocentric")
    args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
    args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
args.save_path = os.path.join(args.save_root, time_str)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
dict_args = vars(args)

shutil.copy('./models/locate.py', args.save_path)
shutil.copy('./train.py', args.save_path)

str_1 = ""
for key, value in dict_args.items():
    str_1 += key + "=" + str(value) + "\n"

logging.basicConfig(filename='%s/run.log' % args.save_path, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info(str_1)

def post_process(KLs, SIM, NSS, ego_pred, GT_mask, args):
    ego_pred = np.array(ego_pred.squeeze().data.cpu())
    ego_pred = normalize_map(ego_pred, args.crop_size)
    kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)
    KLs.append(kld)
    SIM.append(sim)
    NSS.append(nss)
    return KLs, SIM, NSS


if __name__ == '__main__':
    set_seed(seed=0)

    from data.datatrain import TrainData

    trainset = TrainData(exocentric_root=args.exocentric_root,
                         egocentric_root=args.egocentric_root,
                         resize_size=args.resize_size,
                         crop_size=args.crop_size, divide=args.divide)

    TrainLoader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=False)

    from data.datatest import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=False)

    model = model(aff_classes=args.num_classes, args=args)

    model = model.cuda()
    model.train()
    optimizer, scheduler = get_optimizer(model, args)

    best_kld = 1000
    best_epoch = 0
    best_sim=0
    best_nss =0

    best_ref_kld = 1000
    best_ref_epoch = 0
    best_ref_sim = 0
    best_ref_nss = 0

    best_rem_kld = 1000
    best_rem_epoch = 0
    best_rem_sim = 0
    best_rem_nss = 0

    best_m_kld = 1000

    current_iter = 0
    for epoch in range(args.epochs):
        model.train()
        logger.info('LR = ' + str(scheduler.get_last_lr()))
        exo_aff_acc = AverageMeter()
        ego_aff_acc = AverageMeter()
        ego_obj_acc = AverageMeter()


        for step, (exocentric_image, egocentric_image, aff_label, aff_name) in enumerate(TrainLoader):
            aff_label = aff_label.cuda().long()  # b x n x 36
            exo = exocentric_image.cuda()  # b x n x 3 x 224 x 224
            ego = egocentric_image.cuda()

            logits, loss_ce_ego, loss_ce_exo, loss_pixelcont, loss_protocont = \
                model(exo, ego, aff_label, epoch)

            loss_dict = {
                'Lce_eg': loss_ce_ego,
                'Lce_ex': loss_ce_exo,
                'Lcont_pixel': loss_pixelcont,
                'Lcont_proto': loss_protocont
            }

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_batch = exo.size(0)
            exo_acc = 100. * compute_cls_acc(logits['aff_exo'].mean(1), aff_label)
            exo_aff_acc.updata(exo_acc, cur_batch)
            ego_acc = 100. * compute_cls_acc(logits['aff_ego'], aff_label)
            ego_aff_acc.updata(ego_acc, cur_batch)

            if (step + 1) % args.show_step == 0:
                log_str = '%d/%d]%d/%d' % (epoch + 1, args.epochs, step + 1, len(TrainLoader))
                log_str += ' Ac(gx): {:.2f}/{:.2f}'.format(ego_aff_acc.avg, exo_aff_acc.avg)
                log_str += ' Lce(gx): {:.2f}/{:.2f}'.format(loss_ce_ego.item(), loss_ce_exo.item())
                log_str += ' Lpctl: {:.2f}'.format(loss_protocont.item())
                log_str += ' Lctlpx: {:.2f}'.format(loss_pixelcont.item())
                logger.info(log_str)

                current_iter += 1

        scheduler.step()

        KLs, meanKLs = [], []
        SIM, meanSIM = [], []
        NSS, meanNSS = [], []
        reeKLS, remKLS = [], []
        reeSIM, remSIM = [], []
        reeNSS, remNSS = [], []
        model.eval()

        GT_path = args.divide + "_gt.t7"
        if not os.path.exists(GT_path):
            process_gt(args)

        GT_masks = torch.load(args.divide + "_gt.t7")

        for step, (image, label, mask_path) in enumerate(TestLoader):

            cluster_sim_maps = []

            names = mask_path[0].split("/")
            key = names[-3] + "_" + names[-2] + "_" + names[-1]
            GT_mask = GT_masks[key]
            GT_mask = GT_mask / 255.0

            GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))

            ego_pred, refined_CLIP_ego_ego, refined_CLIP_ego_mean = model.test_forward(image.cuda(), label.long().cuda())

            KLs, SIM, NSS = post_process(KLs, SIM, NSS, ego_pred, GT_mask, args)
            reeKLS, reeSIM, reeNSS = post_process(reeKLS, reeSIM, reeNSS, refined_CLIP_ego_ego, GT_mask, args)
            remKLS, remSIM, remNSS = post_process(remKLS, remSIM, remNSS, refined_CLIP_ego_mean, GT_mask, args)

        mKLD = sum(KLs) / len(KLs)
        mSIM = sum(SIM) / len(SIM)
        mNSS = sum(NSS) / len(NSS)

        mreeKLS = sum(reeKLS) / len(reeKLS)
        mreeSIM = sum(reeSIM) / len(reeSIM)
        mreeNSS = sum(reeNSS) / len(reeNSS)

        mremKLS = sum(remKLS) / len(remKLS)
        mremSIM = sum(remSIM) / len(remSIM)
        mremNSS = sum(remNSS) / len(remNSS)

        logger.info(
            "epoch|mKLD|mSIM|mNSS , " + str(epoch + 1) + ", "+ str(round(mKLD, 3)) +", "+ str(round(mSIM, 3)) + ", " + str(round(mNSS, 3)) + ",   "
            + "BEST epoch|mKLD|mSIM|mNSS , " + str(best_epoch + 1) + ", "+ str(round(best_kld, 3)) +", "+ str(round(best_sim, 3)) + ", " + str(round(best_nss, 3)))

        logger.info(
            "refined ego-ego + mKLD|mSIM|mNSS = " + str(round(mreeKLS, 3)) + ", " + str(round(mreeSIM, 3)) + ", " + str(round(mreeNSS, 3)) + ", "
            + "BEST e|mKLD|mSIM|mNSS , " + str(best_ref_epoch + 1) + ", "+ str(round(best_ref_kld, 3)) +", "+ str(round(best_ref_sim, 3)) + ", " + str(round(best_ref_nss, 3)))

        logger.info(
            "refined ego-mean + mKLD|mSIM|mNSS = " + str(round(mremKLS, 3)) + ", " + str(round(mremSIM, 3)) + ", " + str(round(mremNSS, 3)) + ", "
            + "BEST e|mKLD|mSIM|mNSS , " + str(best_rem_epoch + 1) + ", "+ str(round(best_rem_kld, 3)) +", "+ str(round(best_rem_sim, 3)) + ", " + str(round(best_rem_nss, 3)))


        if mKLD < best_kld:
            best_kld = mKLD
            best_epoch = epoch
            best_sim = mSIM
            best_nss = mNSS

        if mreeKLS < best_ref_kld:
            best_ref_kld = mreeKLS
            best_ref_epoch = epoch
            best_ref_sim = mreeSIM
            best_ref_nss = mreeNSS

            model_name = 'best_model_' + str(epoch + 1) + '_' + str(round(mreeKLS, 3)) \
                         + '_' + str(round(mreeSIM, 3)) \
                         + '_' + str(round(mreeNSS, 3)) \
                         + '.pth'
            torch.save(model.state_dict(), os.path.join(args.save_path, model_name))

        if mremKLS < best_rem_kld:
            best_rem_kld = mremKLS
            best_rem_epoch = epoch
            best_rem_sim = mremSIM
            best_rem_nss = mremNSS
