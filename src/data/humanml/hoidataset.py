import codecs as cs
import os.path
import random
import json
from os.path import join as pjoin
from trimesh import Trimesh
import trimesh
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
from .scripts.motion_process import process_file, recover_from_ric
from .scripts.word_vectorizer import WordVectorizer
from .utils import gcn_to_hml3d_format, hml3d_to_gcn_format
import pandas as pd

simplified_mesh = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
    'clothesstand':"clothesstand_cleaned_simplified.obj",
    'floorlamp':"floorlamp_cleaned_simplified.obj",
    'tripod':"tripod_cleaned_simplified.obj",
    'whitechair':"whitechair_cleaned_simplified.obj",
    'woodchair':"woodchair_cleaned_simplified.obj"
}
"""For use of training text-2-motion generative model"""








#behave
class Text2MotionDatasetV2_GCN(data.Dataset):
    def __init__(self, mean, std, split_file, mode,w_vectorizer,max_motion_length,min_motion_length,
                max_text_len,unit_length,motion_dir,text_dir,data_root,use_global,joint_format,repeat_dataset=1):
        
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.min_motion_len = min_motion_length
        self.unit_length=unit_length
        self.motion_dir=motion_dir
        self.text_dir = text_dir
        self.data_root=data_root
        self.max_text_len=max_text_len
        self.use_global=use_global
        self.joint_format=joint_format
        self.repeat_dataset=repeat_dataset
        self.mode=mode
        data_dict = {}
        id_list = []
        #获取所有训练集
        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        tf = open(pjoin(data_root,"text_split.json"), "r")
        self.text_split=json.load(tf)
        loaded_data = np.load(pjoin(data_root,"name_gtmotion.npz"), allow_pickle=True)
        self.name_gtmotion = {key: loaded_data[key] for key in loaded_data.files}
        loaded_data2 = np.load(pjoin(data_root,"name_contact_label.npz"), allow_pickle=True)
        self.name_contact_label = {key: loaded_data2[key] for key in loaded_data2.files}
        
        new_name_list = []
        length_list = []
        #对每个所有训练集，准备文本、物体的信息（单帧）、motion的信息（连续若干长度）
        # print("before read data..", '='*50)

        # for name in tqdm(id_list):
        for name in id_list:
            try:
                contact=np.load(pjoin(data_root,"contact", self.mode ,name+ '.npy'))
                sixD=np.load(pjoin(data_root,"6D", self.mode ,name+ '.npy'))
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                
                # load obj points----------------
                obj_name = name.split('_')[2]
                obj_path = pjoin(data_root, 'object_mesh')
                mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])
                temp_simp = trimesh.load(mesh_path)#是一个对象，存放vec和face的信息

                obj_points = np.array(temp_simp.vertices)#（485,3）
                obj_faces = np.array(temp_simp.faces)#（933,3）
                
                # signed_distances = trimesh.proximity.signed_distance(temp_simp, obj_points)
                

                # center the meshes
                center = np.mean(obj_points, 0)
                obj_points -= center
                obj_points = obj_points.astype(np.float32)


                # sample object points
                ## downsampling idxs for 20 objects
                
                
                obj_sample_path = pjoin(data_root, 'object_sample/{}.npy'.format(name))
                o_choose = np.load(obj_sample_path)#(512,)       
                obj_points_chosen = obj_points[o_choose]
                obj_normals = obj_faces[o_choose] 
                # TODO: hardcode
                motion = motion[:199].astype(np.float32)#(199,269)


                # contact_input = np.load(pjoin(opt.data_root, 'affordance_data/contact_'+name + '.npy'), allow_pickle=True)[None][0]
                
                if (len(motion)) < self.min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    #每个样本不止一个文本
                    
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:

                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < self.min_motion_len or (len(n_motion) >= 200):
                                    continue
                                # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
        
                                data_dict[name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict],
                                                    'seq_name': name,
                                                    'obj_points': obj_points_chosen,
                                                    'obj_normals':obj_normals,
                                                    'contact':contact,
                                                    'sixD':sixD,
                                                    "temp_simp":obj_faces,
                                                    "all_obj_points":obj_points
                                                    # 'gt_afford_labels': contact_input
                                                }
                                new_name_list.append(name)
                                length_list.append(len(n_motion))
                            except:
                                continue

                if flag:
                    data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data,
                                        'seq_name': name,
                                        'obj_points': obj_points_chosen,
                                        'obj_normals':obj_normals,
                                        'contact':contact,
                                        'sixD':sixD,
                                        "temp_simp":obj_faces,
                                        "all_obj_points":obj_points
                                    }

                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as err:
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.prepare()
        
    def prepare(self):
        skeleton = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        for key in self.data_dict:
            motion = (self.data_dict[key]["motion"] - self.mean) / self.std
            contact=self.data_dict[key]["contact"]
            sixD=self.data_dict[key]["sixD"]
            #contact_point=[0,10,11,12,16,17,20,21]
            contact_pro = np.ones((contact.shape[0], contact.shape[1]))
            #contact_pro = np.ones((contact.shape[0], len(contact_point)))
            contact_pro = -1 * contact_pro
            

            dis = {"backpack":0.10,'boxlarge':0.17,'boxlong':0.17,'boxmedium':0.17, 'boxsmall':0.17, 'boxtiny':0.17, 'chairblack':0.15,
            'chairwood':0.15, 'monitor':0.15, 'plasticcontainer':0.15, 'stool':0.12, 'suitcase':0.15, 'tablesmall':0.12, 
            'tablesquare':0.1, 'toolbox':0.1, 'trashbin':0.14, 'yogaball':0.12, 'yogamat':0.1}

            contact=contact <= dis[self.data_dict[key]["seq_name"].split("_")[2]]
            contact_pro[contact] = 1

            self.data_dict[key]["contact_pro"]=contact_pro
            motion = np.concatenate((motion,sixD), axis=1)
            self.data_dict[key]["gcn_motion"] = hml3d_to_gcn_format(motion, self.joint_format)
            
    def to_hml3d_format(self, data):
        return gcn_to_hml3d_format(data, self.joint_format)

    def to_gcn_format(self, data):
        return hml3d_to_gcn_format(data, self.joint_format)
        
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        data = data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
        return data

    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(
            data.device) + torch.from_numpy(self.mean).to(data.device)
        return data
    
    def norm_transform_th(self,data):
        data = (data - torch.from_numpy(self.mean).to(data.device))/torch.from_numpy(self.std).to(data.device)
        return data



    def __len__(self):
        return (len(self.data_dict) - self.pointer) * self.repeat_dataset

    def __getitem__(self, item):
        
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx % len(self.data_dict)]]#data_name
        motion,gcn_motion, m_length, text_list, seq_name, obj_points, obj_normals =data["motion"], data['gcn_motion'], data['length'], data['text'], data['seq_name'],  data['obj_points'], data['obj_normals']
        contact,contact_pro,mesh,all_obj_points = data['contact'],data['contact_pro'],data["temp_simp"],data["all_obj_points"]
        sixD = data["sixD"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        #self.opt.max_text_len:20
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:

            if len(token.split('/'))<2:
                print(f" {seq_name}   {tokens}")
                break
            #word_emb:(300,)：text单词编码
            #pos_oh(15,)：词性
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)#(len,15)
        word_embeddings = np.concatenate(word_embeddings, axis=0)#(len,300)

        # Crop the motions in to times of 4, and introduce small variations
        #不再是199长度的motion，将长度变为4的倍数
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        
        idx = random.randint(0, len(gcn_motion) - m_length)
        #idx = 0
        gcn_motion = gcn_motion[idx:idx+m_length]
        contact_pro=contact_pro[idx:idx+m_length]
        contact=contact[idx:idx+m_length]
        motion=motion[idx:idx+m_length]
        sixD=sixD[idx:idx+m_length]

        # self.name_gtmotion[seq_name]=self.name_gtmotion[seq_name][0][idx:idx+m_length]
        # self.name_contact_label[seq_name]=self.name_contact_label[seq_name][idx:idx+m_length]
        return gcn_motion,m_length,caption,word_embeddings,pos_one_hots, sent_len, obj_points, obj_normals, seq_name,contact,contact_pro,mesh,all_obj_points,motion,sixD,idx
        #return result






      





class Text2MotionOmomoDatasetV2_GCN(data.Dataset):
    def __init__(self, mode,mean, std, split_file, w_vectorizer,max_motion_length,min_motion_length,
                max_text_len,unit_length,motion_dir,text_dir,njoints,data_root,use_global,joint_format,repeat_dataset=1):
        
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.min_motion_len = min_motion_length
        self.unit_length=unit_length
        self.motion_dir=motion_dir
        self.text_dir = text_dir
        self.data_root=data_root
        self.max_text_len=max_text_len
        self.use_global=use_global
        self.joint_format=joint_format
        self.repeat_dataset=repeat_dataset
        self.mode=mode
        data_dict = {}
        id_list = []
        #获取所有训练集

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        tf = open("./dataset/omomo/text_split.json", "r")
        self.text_split=json.load(tf)
        loaded_data = np.load("./dataset/omomo/name_gtmotion.npz", allow_pickle=True)
        self.name_gtmotion = {key: loaded_data[key] for key in loaded_data.files}
        loaded_data2 = np.load("./dataset/omomo/name_contact_label.npz", allow_pickle=True)
        self.name_contact_label = {key: loaded_data2[key] for key in loaded_data2.files}
        

        new_name_list = []
        length_list = []
        
        #对每个所有训练集，准备文本、物体的信息（单帧）、motion的信息（连续若干长度）
        
        tf = open("./dataset/omomo_t2m_final_nofacez/obj_scale.json", "r")
        obj_scale_list = json.load(tf)
        for name in tqdm(id_list):
            try:
                sixD=np.load(pjoin("./dataset/omomo_t2m_final_nofacez/6d", self.mode ,name+ '.npy'))
                contact=np.load(pjoin("./dataset/omomo_t2m_final_nofacez/contact", self.mode ,name+ '.npy'))
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                # # load gt obj points----------------
                obj_name = name.split('_')[1]
                # load obj points----------------
                obj_mesh_path = pjoin('/home/guohong/omomo/data/capture_objects_simplify', obj_name+"_cleaned_simplified.obj")
                mesh = trimesh.load_mesh(obj_mesh_path)
                obj_scale=obj_scale_list[str(obj_name)]
                obj_points = np.array(mesh.vertices)#（485,3）
                obj_faces = np.array(mesh.faces)#（933,3）
                obj_points *= obj_scale
                # center the meshes
                center = np.mean(obj_points, 0)
                obj_points -= center
                obj_points = obj_points.astype(np.float32)
                # TODO: hardcode
                motion = motion[:199].astype(np.float32)#(199,269)
                sixD = sixD[:199].astype(np.float32)#(199,269)
                contact = contact[:199].astype(np.float32)
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    #每个样本不止一个文本
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                data_dict[name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict],
                                                    'seq_name': name,
                                                    'gtobj_points': None,
                                                    "gttemp_simp":None,
                                                    "obj_scale":obj_scale,
                                                    'sixD':sixD,
                                                    'contact':contact,
                                                    "obj_points":obj_points,
                                                    "temp_simp":obj_faces,
                                                }
                                new_name_list.append(name)
                                length_list.append(len(n_motion))
                            except:
                                continue
                
                if flag:
                    data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data,
                                        'seq_name': name,
                                        'gtobj_points': None,
                                        "gttemp_simp":None,
                                        "obj_scale":obj_scale,
                                        'sixD':sixD,
                                        'contact':contact,
                                        "obj_points":obj_points,
                                        "temp_simp":obj_faces,
                                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as err:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.prepare()
        
    def prepare(self):
        skeleton = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        for key in self.data_dict:
            motion = (self.data_dict[key]["motion"] - self.mean) / self.std
            contact=self.data_dict[key]["contact"]
            sixD=self.data_dict[key]["sixD"]
            contact_pro = np.zeros((contact.shape[0], contact.shape[1]))
            contact = contact <= 0.5
            contact_pro[contact] = 1
            self.data_dict[key]["contact_pro"]=contact_pro
            motion = np.concatenate((motion,sixD), axis=1)
            self.data_dict[key]["gcn_motion"] = hml3d_to_gcn_format(motion, self.joint_format)
            
    def to_gcn_format(self, data):
        return hml3d_to_gcn_format(data, self.joint_format)
            
    def to_hml3d_format(self, data):
        return gcn_to_hml3d_format(data, self.joint_format)
        
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        data = data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
        return data

    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(
            data.device) + torch.from_numpy(self.mean).to(data.device)
        return data


    def __len__(self):
        return (len(self.data_dict) - self.pointer) * self.repeat_dataset

    def __getitem__(self, item):

        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx % len(self.data_dict)]]#data_name
        motion,gcn_motion, m_length,text_list, seq_name, obj_points,faces = data['motion'], data['gcn_motion'],data['length'], data['text'], data['seq_name'],  data['obj_points'],data["temp_simp"]
        sixD=data["sixD"]
        #gtobj_points,gtfaces=data['gtobj_points'],data["gttemp_simp"]
        obj_scale=data["obj_scale"]
        contact = data['contact']
        contact_pro = data["contact_pro"]
        # # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        
        #self.opt.max_text_len:20
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:

            if len(token.split('/'))<2:
                print(f" {seq_name}   {tokens}")
                break
            #word_emb:(300,)：text单词编码
            #pos_oh(15,)：词性
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)#(len,15)
        word_embeddings = np.concatenate(word_embeddings, axis=0)#(len,300)
        
        # Crop the motions in to times of 4, and introduce small variations
        #不再是199长度的motion，将长度变为4的倍数

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single','double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        #m_length = (m_length // self.unit_length - 1) * self.unit_length
        #idx = random.randint(0, len(motion) - m_length)
        idx = 0
        motion = motion[idx:idx+m_length]
        contact=contact[idx:idx+m_length]
        contact_pro=contact_pro[idx:idx+m_length]
        gcn_motion = gcn_motion[idx:idx+m_length]
        sixD = sixD[idx:idx+m_length]
        return gcn_motion,m_length,caption,word_embeddings,pos_one_hots, sent_len, seq_name,contact,obj_scale,motion,obj_points,faces,contact_pro,sixD,idx



