# import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from datasets import VITONDataset
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_opt():
    class Config:
        def __init__(self):
            self.name = 'default_name'           
            self.batch_size = 1
            self.workers = 1
            self.load_height = 1024
            self.load_width = 768
            self.shuffle = False  
            self.dataset_dir = './datasets/'
            self.dataset_mode = 'test'
            self.checkpoint_dir = './checkpoints/'
            self.save_dir = './datasets/results/'
            self.display_freq = 1

            self.seg_checkpoint = 'seg_final.pth'
            self.gmm_checkpoint = 'gmm_final.pth'
            self.alias_checkpoint = 'alias_final.pth'

            # common
            self.semantic_nc = 13
            self.init_type = 'xavier'
            self.init_variance = 0.02

            # for GMM
            self.grid_size = 5

            # for ALIASGenerator
            self.norm_G = 'spectralaliasinstance'
            self.ngf = 64
            self.num_upsampling_layers = 'most'
    
    return Config()


def infer(opt, seg, gmm, alias,img_name=None, c_name=None):
    seg.to(device)
    gmm.to(device)
    alias.to(device)

    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)

    test_dataset = VITONDataset(opt,img_name=img_name, c_name=c_name)
    inputs = test_dataset[0]

    img_name = inputs['img_name']
    c_name = inputs['c_name']

    img_agnostic = inputs['img_agnostic'].to(device).unsqueeze(0)
    parse_agnostic = inputs['parse_agnostic'].to(device).unsqueeze(0)
    pose = inputs['pose'].to(device).unsqueeze(0)
    c = inputs['cloth'].to(device).unsqueeze(0)
    cm = inputs['cloth_mask'].to(device).unsqueeze(0)

    # Part 1. Segmentation generation
    parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
    pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
    c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
    cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
    seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).to(device)), dim=1)

    parse_pred_down = seg(seg_input)
    parse_pred = gauss(up(parse_pred_down))
    parse_pred = parse_pred.argmax(dim=1)[:, None]

    parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).to(device)
    parse_old.scatter_(1, parse_pred, 1.0)

    labels = {
        0:  ['background',  [0]],
        1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
        2:  ['upper',       [3]],
        3:  ['hair',        [1]],
        4:  ['left_arm',    [5]],
        5:  ['right_arm',   [6]],
        6:  ['noise',       [12]]
    }
    parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).to(device)
    for j in range(len(labels)):
        for label in labels[j][1]:
            parse[:, j] += parse_old[:, label]

    # Part 2. Clothes Deformation
    agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
    parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
    pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
    c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
    gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

    _, warped_grid = gmm(gmm_input, c_gmm)
    warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
    warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

    # Part 3. Try-on synthesis
    misalign_mask = parse[:, 2:3] - warped_cm
    misalign_mask[misalign_mask < 0.0] = 0.0
    parse_div = torch.cat((parse, misalign_mask), dim=1)
    parse_div[:, 2:3] -= misalign_mask

    output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

    save_image(output, img_name, c_name, opt.save_dir)
    print("Image synthesized successfully.")


def main(img_name=None, c_name=None):
    opt = get_opt()
    # print(opt)

    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc).to(device)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3).to(device)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9).to(device)
    opt.semantic_nc = 13
    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))
    infer(opt, seg, gmm, alias,img_name=img_name, c_name=c_name)


if __name__ == '__main__':
    main()
