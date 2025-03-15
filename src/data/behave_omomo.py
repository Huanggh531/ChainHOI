# -*- coding: utf-8 -*-
"""
 @File    : behave.py
 @Time    : 2023/3/27 20:23
 @Author  : Ling-An Zeng
 @Email   : linganzeng@gmail.com
 @Software: PyCharm
"""
from typing import Optional, Callable
import os.path as osp
from os.path import join as pjoin
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from argparse import Namespace
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
# from lightning_utilities.core.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from .utils import behaveGCN_collate,WordVectorizer,omomoGCN_collate

from .humanml.hoidataset import Text2MotionDatasetV2_GCN,Text2MotionOmomoDatasetV2_GCN




class BehaveGCNDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_batch_size: int = -1,
        test_batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = False,
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        max_text_len: int = 20,
        unit_length: int = 4,
        w_vectorizer_path: str = 'glove',
        dataset_name: str = 'hml3d',
        data_root:str = 'dataset/behave_t2m',
        use_global:bool=False,
        joint_format:str='',
        repeat_dataset:int=1
    ):
        super().__init__()
        self.batch_size=batch_size
        self.val_batch_size=val_batch_size
        self.test_batch_size=test_batch_size
        self.max_motion_length=max_motion_length
        self.min_motion_length=min_motion_length
        self.max_text_len=max_text_len
        self.unit_length=unit_length
        self.save_hyperparameters(logger=False)
        # Configurations of T2M dataset and KIT dataset is almost the same
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        print('Loading dataset %s ...' % dataset_name)
        self.w_vectorizer = WordVectorizer(w_vectorizer_path, 'our_vab')#"WordVectorizer"可能是指用于将文本数据转换为向量表示
        #self.dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
        self.dataset = Text2MotionDatasetV2_GCN
        self.dataloader_options = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": False,
            "collate_fn": behaveGCN_collate
        }
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_root=data_root
        self.use_global=use_global
        self.joint_format=joint_format
        self.repeat_dataset=repeat_dataset


    def setup(self,stage:None):
        #self.hparams.keyframe_info.file = osp.join(self.data_dir, self.hparams.keyframe_info.file)
        self.dataset_kwargs = {
            "mean": np.load(osp.join(self.data_root, "Mean_local.npy")),
            "std": np.load(osp.join(self.data_root, "Std_local.npy")),
            "w_vectorizer": self.w_vectorizer,
            "max_motion_length": self.max_motion_length,
            "min_motion_length": self.min_motion_length,
            "max_text_len": self.max_text_len,
            "unit_length": self.unit_length,
            "motion_dir": osp.join(self.data_root, "new_joint_vecs_local"),
            "text_dir": osp.join(self.data_root, "texts"),
            "data_root":self.data_root,
            "use_global":self.use_global,
            "joint_format":self.joint_format,
            
        }

    def train_dataloader(self):
        
        if self.train_dataset is None:
            self.train_dataset = self.dataset(split_file=osp.join(self.data_root, "train.txt"),mode="train",
                                                      repeat_dataset=self.repeat_dataset,**self.dataset_kwargs)
            #self.nfeats = self.train_dataset.nfeats
        options = self.dataloader_options.copy()
        options["batch_size"] = self.batch_size
        return DataLoader(dataset=self.train_dataset, shuffle=True, **options)

    def val_dataloader(self):
        
        if self.val_dataset is None:
            self.val_dataset = self.dataset(split_file=osp.join(self.data_root, "test.txt"),mode="test",
                                                            **self.dataset_kwargs)
        options = self.dataloader_options.copy()
        options["batch_size"] = self.val_batch_size
        # if options["batch_size"] == -1:
        #     options["batch_size"] = self.hparams.batch_size
        return DataLoader(dataset=self.val_dataset, shuffle=False, drop_last=False, **options)

    def test_dataloader(self):
        
        if self.test_dataset is None:
            self.test_dataset = self.dataset(split_file=osp.join(self.data_root, "test.txt"),mode="test",
                                                     **self.dataset_kwargs)
            #self.nfeats = self.test_dataset.nfeats
            #self.test_dataset.is_mm = False

        options = self.dataloader_options.copy()
        options["batch_size"] = self.test_batch_size

        return DataLoader(dataset=self.test_dataset, shuffle=False, drop_last=False, **options)
    
    def test_dataloader2(self):
        
        if self.test_dataset is None:
            self.test_dataset = self.dataset(split_file=osp.join(self.data_root, "test_time.txt"),mode="test",
                                                     **self.dataset_kwargs)
            #self.nfeats = self.test_dataset.nfeats
            #self.test_dataset.is_mm = False

        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.test_batch_size

        return DataLoader(dataset=self.test_dataset, shuffle=False, drop_last=False, **options)



    def mm_mode(self, mm_on=True, mm_num_samples=100):
        # random select samples for mm
        if mm_on:
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list, mm_num_samples, replace=False)
            self.test_dataset.name_list = self.mm_list
            self.test_dataset.is_mm = True
        else:
            self.test_dataset.is_mm = False
            self.test_dataset.name_list = self.name_list



class OmomoDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        val_batch_size: int = -1,
        test_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        max_text_len: int = 20,
        unit_length: int = 4,
        njoints: int = 22,
        w_vectorizer_path: str = 'glove',
        dataset_name: str = 'hml3d',
        data_root:str = 'dataset/omomo',
        use_global:bool=False,
        joint_format:str='',
        repeat_dataset:int=1
    ):
        super().__init__()
        self.batch_size=batch_size
        self.val_batch_size=val_batch_size
        self.test_batch_size=test_batch_size
        self.max_motion_length=max_motion_length
        self.min_motion_length=min_motion_length
        self.max_text_len=max_text_len
        self.unit_length=unit_length
        self.save_hyperparameters(logger=False)
        # Configurations of T2M dataset and KIT dataset is almost the same
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        print('Loading dataset %s ...' % dataset_name)
        self.w_vectorizer = WordVectorizer(w_vectorizer_path, 'our_vab')#"WordVectorizer"可能是指用于将文本数据转换为向量表示
        #self.dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
        self.dataset = Text2MotionOmomoDatasetV2_GCN
        self.dataloader_options = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": False,
            "collate_fn": omomoGCN_collate
        }
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_root=data_root
        self.njoints=njoints
        self.use_global=use_global
        self.joint_format=joint_format
        self.repeat_dataset=repeat_dataset


    def setup(self,stage:None):
        #self.hparams.keyframe_info.file = osp.join(self.data_dir, self.hparams.keyframe_info.file)

        self.dataset_kwargs = {
            "mean": np.load(osp.join(self.data_root, "Mean_local.npy")),
            "std": np.load(osp.join(self.data_root, "Std_local.npy")),
            "w_vectorizer": self.w_vectorizer,
            "max_motion_length": self.max_motion_length,
            "min_motion_length": self.min_motion_length,
            "max_text_len": self.max_text_len,
            "unit_length": self.unit_length,
            "motion_dir": osp.join(self.data_root, "new_joint_vecs_local"),
            "text_dir": osp.join(self.data_root, "texts"),
            "njoints": self.njoints,
            "data_root":self.data_root,
            "use_global":self.use_global,
            "joint_format":self.joint_format,
            
        }

    def train_dataloader(self):
        
        if self.train_dataset is None:
            self.train_dataset = self.dataset(mode="train",split_file=osp.join(self.data_root, "train.txt"),
                                                      repeat_dataset=self.repeat_dataset,**self.dataset_kwargs)
            
            #self.nfeats = self.train_dataset.nfeats
        
        options = self.dataloader_options.copy()
        options["batch_size"] = self.batch_size
        return DataLoader(dataset=self.train_dataset, shuffle=True, **options)

    def val_dataloader(self):
        
        if self.val_dataset is None:
            self.val_dataset = self.dataset(mode="test",split_file=osp.join(self.data_root, "test.txt"),
                                                            **self.dataset_kwargs)
        options = self.dataloader_options.copy()
        options["batch_size"] = self.val_batch_size
        # if options["batch_size"] == -1:
        #     options["batch_size"] = self.hparams.batch_size
        return DataLoader(dataset=self.val_dataset, shuffle=False, drop_last=False, **options)

    def test_dataloader(self):
        
        if self.test_dataset is None:
            self.test_dataset = self.dataset(mode="test",split_file=osp.join(self.data_root, "test.txt"),
                                                     **self.dataset_kwargs)
            #self.nfeats = self.test_dataset.nfeats
            #self.test_dataset.is_mm = False

        options = self.dataloader_options.copy()
        options["batch_size"] = self.test_batch_size

        return DataLoader(dataset=self.test_dataset, shuffle=False, drop_last=False, **options)


    def mm_mode(self, mm_on=True, mm_num_samples=100):
        # random select samples for mm
        if mm_on:
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list, mm_num_samples, replace=False)
            self.test_dataset.name_list = self.mm_list
            self.test_dataset.is_mm = True
        else:
            self.test_dataset.is_mm = False
            self.test_dataset.name_list = self.name_list






if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    datamodule = OmomoDataModule("./data",joint_format="behave_joint_5")
    datamodule.setup(None)
    dataloader = datamodule.train_dataloader()
    for i, data in enumerate(dataloader):
        # motion, text, length, word_embs, pos_ohot, text_len, tokens = data
        print(data["motion"].shape, data["text"], data["length"], data["text_len"])
        break