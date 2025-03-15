import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from .motion_clip import MotionCLIP
from .motion_clip import MotionCLIP2

class MotionCLIPEvaluator(object):
    def __init__(self, dataset='hml3d', deps_dir="./deps/MotionCLIP"):
        self.dataset = dataset
        self.motion_clip = MotionCLIP(motion_dim=263 if self.dataset == 'hml3d' else 263)
        
        #ckpt = torch.load(osp.join(deps_dir, f'{self.dataset}/{self.dataset}.ckpt'), map_location='cpu')

        ckpt=torch.load("checkpoint/behave_t2m/epoch=24-step=325.ckpt",map_location='cpu')
        # self.motion_clip.load_state_dict(ckpt, strict=True)
        new_ckpt={}
        
        # self.motion_clip.load_state_dict(ckpt, strict=True)
        for name,param in ckpt['state_dict'].items():
            new_ckpt[name.replace("motion_clip.","")]=param

        self.motion_clip.load_state_dict(new_ckpt)
        #self.mean = torch.tensor(np.load(osp.join(deps_dir, f'{self.dataset.split("_")[1]}_mean.npy')), dtype=torch.float)
        #self.std = torch.tensor(np.load(osp.join(deps_dir, f'{self.dataset.split("_")[1]}_std.npy')), dtype=torch.float)

    def extract_embedding(self, motion, motion_len, text):
        text_embed = self.extract_text_embedding(motion.device, text)
        motion_embed = self.extract_motion_embedding(motion, motion_len)
        return motion_embed, text_embed

    @torch.no_grad()
    def extract_text_embedding(self, device, text):
        self.motion_clip.to(device)
        self.motion_clip.eval()
        text_embed = self.motion_clip.encode_text(text, device).float()
        return text_embed / text_embed.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def extract_motion_embedding(self, motion, motion_len):

        self.motion_clip.to(motion.device)
        self.motion_clip.eval()
        device = motion.device
        #motion = (motion - self.mean.to(device)) / self.std.to(device)
        motion = motion.float()
        
        motion_embed = self.motion_clip.encode_motion(motion, motion_len).type(motion.dtype)
        # return motion_embed
        return motion_embed / motion_embed.norm(dim=1, keepdim=True)


class MotionCLIPEvaluator2(object):
    def __init__(self, dataset='omomo', deps_dir="./deps/MotionCLIP"):
        self.dataset = dataset
        self.motion_clip = MotionCLIP2(motion_dim=263 if self.dataset == 'omomo' else 263)
        
        #ckpt = torch.load(osp.join(deps_dir, f'{self.dataset}/{self.dataset}.ckpt'), map_location='cpu')

        ckpt=torch.load("checkpoint/omomo/epoch=273-step=9590.ckpt",map_location='cpu')
        # self.motion_clip.load_state_dict(ckpt, strict=True)
        new_ckpt={}
        
        # self.motion_clip.load_state_dict(ckpt, strict=True)
        for name,param in ckpt['state_dict'].items():
            new_ckpt[name.replace("motion_clip.","")]=param

        self.motion_clip.load_state_dict(new_ckpt)
        #self.mean = torch.tensor(np.load(osp.join(deps_dir, f'{self.dataset.split("_")[1]}_mean.npy')), dtype=torch.float)
        #self.std = torch.tensor(np.load(osp.join(deps_dir, f'{self.dataset.split("_")[1]}_std.npy')), dtype=torch.float)

    def extract_embedding(self, motion, motion_len, text):
        text_embed = self.extract_text_embedding(motion.device, text)
        motion_embed = self.extract_motion_embedding(motion, motion_len)
        return motion_embed, text_embed

    @torch.no_grad()
    def extract_text_embedding(self, device, text):
        self.motion_clip.to(device)
        self.motion_clip.eval()
        text_embed = self.motion_clip.encode_text(text, device).float()
        return text_embed / text_embed.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def extract_motion_embedding(self, motion, motion_len):

        self.motion_clip.to(motion.device)
        self.motion_clip.eval()
        device = motion.device
        #motion = (motion - self.mean.to(device)) / self.std.to(device)
        motion = motion.float()
        
        motion_embed = self.motion_clip.encode_motion(motion, motion_len).type(motion.dtype)
        # return motion_embed
        return motion_embed / motion_embed.norm(dim=1, keepdim=True)
