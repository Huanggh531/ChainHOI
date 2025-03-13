import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F

# padding to max length in one batch
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


# an adapter to our collate func

def gcn_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {
        "gcn_motion": collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "motion_len": collate_tensors([torch.tensor(b[1]) for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "word_embs": collate_tensors([torch.tensor(b[3]).float() for b in notnone_batches]),
        "pos_ohot": collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        "text_len": collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
    }
    return adapted_batch


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas
    
def collate_tensors2(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        pad_sizes = []
        for d in range(dims - 1, -1, -1):
            pad_sizes.extend([0, max_size[d] - b.size(d)])
        # 获取最后一个点的坐标值
        last_point = b[-1, :].unsqueeze(0)  # shape [1, 3]
        padding_value = last_point.repeat((max_size[0] - b.size(0), 1))  # shape [pad_size, 3]
        
        # 填充点云数据
        padded_b = torch.cat([b, padding_value], dim=0)
        
        canvas[i, :padded_b.size(0), :padded_b.size(1)] = padded_b
    return canvas




#behave_gcn结构
def behaveGCN_collate(batch):

    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {
        
        "gcn_motion": collate_tensors([torch.tensor(b[0],dtype=torch.float) for b in notnone_batches]),
        "motion_len": collate_tensors([torch.tensor(b[1]) for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "word_embs": collate_tensors([torch.tensor(b[3],dtype=torch.float) for b in notnone_batches]),
        "pos_ohot": collate_tensors([torch.tensor(b[4],dtype=torch.float) for b in notnone_batches]),
        "text_len": collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
        "obj_points":collate_tensors([torch.tensor(b[6]) for b in notnone_batches]),
        "obj_normals":collate_tensors([torch.tensor(b[7]) for b in notnone_batches]),
        "seq_name":[b[8] for b in notnone_batches],
        "contact":collate_tensors([torch.tensor(b[9]) for b in notnone_batches]),
        "contact_pro":collate_tensors([torch.tensor(b[10]) for b in notnone_batches]),
        #"contact_pro":[torch.tensor(b[10]) for b in notnone_batches],
        "faces": [torch.tensor(b[11]).long() for b in notnone_batches],
        "all_obj_points":collate_tensors2([torch.tensor(b[12]) for b in notnone_batches]),
        "motion_unorm_gt":collate_tensors([torch.tensor(b[13],dtype=torch.float) for b in notnone_batches]),
        "sixD":collate_tensors([torch.tensor(b[14],dtype=torch.float) for b in notnone_batches]),
        "idx": [b[15] for b in notnone_batches],
    }
    return adapted_batch










def omomoGCN_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {
        
        "gcn_motion": collate_tensors([torch.tensor(b[0],dtype=torch.float) for b in notnone_batches]),
        "motion_len": collate_tensors([torch.tensor(b[1]) for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "word_embs": collate_tensors([torch.tensor(b[3],dtype=torch.float) for b in notnone_batches]),
        "pos_ohot": collate_tensors([torch.tensor(b[4],dtype=torch.float) for b in notnone_batches]),
        "text_len": collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
        #"gt_obj_points":collate_tensors([torch.tensor(b[6]) for b in notnone_batches]),
        #"obj_normals":collate_tensors([torch.tensor(b[7]) for b in notnone_batches]),
        "seq_name":[b[6] for b in notnone_batches],
        #"contact":collate_tensors([torch.tensor(b[9]) for b in notnone_batches]),
        "contact":collate_tensors([torch.tensor(b[7]) for b in notnone_batches]),
        #"gt_faces": [torch.tensor(b[9]).long() for b in notnone_batches],
        "obj_scale": [b[8] for b in notnone_batches],
        "motion_unorm_gt":collate_tensors2([torch.tensor(b[9]) for b in notnone_batches]),
        "obj_points":collate_tensors([torch.tensor(b[10]) for b in notnone_batches]),
        "faces": [torch.tensor(b[11]).long() for b in notnone_batches],
        "contact_pro":collate_tensors([torch.tensor(b[12]) for b in notnone_batches]),
        "sixD":collate_tensors([torch.tensor(b[13],dtype=torch.float) for b in notnone_batches]),
        "idx": [b[14] for b in notnone_batches],


    }
    return adapted_batch






import numpy as np
import pickle
from os.path import join as pjoin

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball', 'trashbin', 
            'yogaball', 'yogamat', 'suitcase', 'tablesmall', 'tablesquare', 'backpack', 'boxlong', 'boxsmall', 'boxtiny', 
            'boxlarge', 'boxmedium', 'plasticcontainer', 'stool', 'toolbox', 'monitor', 'chairwood', 'chairblack' )

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb', 'hold')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))#(4199,300)
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))#(4199)
        word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec

