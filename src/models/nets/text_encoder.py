import torch
import torch as th
import torch.nn as nn
from typing import List, Dict

import clip
from transformers import RobertaTokenizer, RobertaModel, logging
logging.set_verbosity_error()

from ..utils.utils import lengths_to_mask


class Roberta(torch.nn.Module):
    def __init__(self, freeze_lm=True, max_text_len=40):
        super(Roberta, self).__init__()
        #self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        #self.lm = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("/home/guohong/pmg-pro/src/models/nets/pretrain/")
        self.lm = RobertaModel.from_pretrained("/home/guohong/pmg-pro/src/models/nets/pretrain")
        self.max_text_len = max_text_len
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

    def forward(self, text, device, **kwargs):
        # 设置最小值为2，防止报错，防止一个token时decoder输出为nan
        #text_len = max(min(min([len(x) for x in text]), self.max_text_len), 2) + 1
        text_len = max([len(x) for x in text])
       
        #text_len=self.max_text_len
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=text_len,
        ).to(device)
        out = self.lm(**encoded_input)

        mask = lengths_to_mask(encoded_input['attention_mask'].argmin(dim=-1), device)
        if mask.shape[1] < out["last_hidden_state"].shape[1]:
            out["last_hidden_state"] = out["last_hidden_state"][:, :mask.size(1)]
        # return {"text_emb": out["pooler_output"], "hidden": out["last_hidden_state"],
        #         "mask": encoded_input["attention_mask"].bool()}
        return {"text_emb": out["pooler_output"], "hidden": out["last_hidden_state"], "mask": mask}


class CLIP(torch.nn.Module):
    def __init__(self, freeze_lm=True):
        super(CLIP, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=torch.device('cpu'), jit=False)
        self.clip_model.visual = None
        # clip.model.convert_weights(self.clip_model)
        if freeze_lm:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    @property
    def dtype(self):
        return self.clip_model.text_projection.dtype

    def forward(self, text, device, **kwargs):
        # import pdb
        # pdb.set_trace()
        text = clip.tokenize(text, truncate=True).to(device)
        x = self.clip_model.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        mask = lengths_to_mask(text.argmax(dim=-1), device)
        if mask.shape[1] < x.shape[1]:
            x = x[:, :mask.size(1)]
        return {"text_emb": text_embed, "hidden": x, "mask": mask}

    def create_masks(self, text):
        objmask_list = []
        humanmask_list = []

        # 假设我们已知要寻找的物体和人类的名称列表
        objects = ['chair', 'box', 'table', 'suitcase']  # 根据需要修改
        humans = ['person', 'individual', 'man', 'woman']  # 根据需要修改

        for sentence in text:
            words = sentence.split()
            objmask = torch.zeros(len(words), dtype=torch.float32)
            humanmask = torch.zeros(len(words), dtype=torch.float32)

            # 标记物体的位置
            for obj in objects:
                for idx, word in enumerate(words):
                    if obj in word.lower():
                        objmask[idx] = 1.0  # 设置为1，表示该位置是一个物体

            # 标记人类的位置
            for human in humans:
                for idx, word in enumerate(words):
                    if human in word.lower():
                        humanmask[idx] = 1.0  # 设置为1，表示该位置是一个人类

            objmask_list.append(objmask)
            humanmask_list.append(humanmask)

        # 将每个句子的掩码堆叠成一个张量
        objmask_tensor = torch.stack(objmask_list).to(text.device)
        humanmask_tensor = torch.stack(humanmask_list).to(text.device)

        # 确保两个掩码的维度相同
        max_length = max(objmask_tensor.size(1), humanmask_tensor.size(1))

        if objmask_tensor.size(1) < max_length:
            padding = torch.zeros(objmask_tensor.size(0), max_length - objmask_tensor.size(1), dtype=objmask_tensor.dtype).to(text.device)
            objmask_tensor = torch.cat([objmask_tensor, padding], dim=1)

        if humanmask_tensor.size(1) < max_length:
            padding = torch.zeros(humanmask_tensor.size(0), max_length - humanmask_tensor.size(1), dtype=humanmask_tensor.dtype).to(text.device)
            humanmask_tensor = torch.cat([humanmask_tensor, padding], dim=1)

        return objmask_tensor, humanmask_tensor