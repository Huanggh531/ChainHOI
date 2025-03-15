'''
使用gcn-v1

'''


import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

from .utils.gcn import unit_gcn
from .utils.tcn import unit_tcn, mstcn
from .utils.graph import Graph

from src.models.utils.embedding import timestep_embedding, TimestepEmbedding, PositionEmbedding


class GCNBase(nn.Module):
    def __init__(
            self,
            layout="hml3d_v1",
            joint_dim=12,
            text_feat_dim=512,
            time_embed_dim=256,
            num_layers: int = 6,
            base_dim: int = 64,
            up_layers: str = "2,4",
            gp_norm: int = 4,
            pos_emb: str = 'cos',
            arch: int = 1,
            input_proj_type: int = 0,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.arch = arch
        self.text_feat_dim = text_feat_dim
        self.num_layers = num_layers
        self.input_proj_type = input_proj_type
        self.time_emb = TimestepEmbedding(time_embed_dim, time_embed_dim)
        self.graph = Graph(layout=layout, mode='spatial')
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        if self.input_proj_type == 0:
            self.input_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.input_norm = nn.GroupNorm(gp_norm, base_dim)
            # self.input_norm = nn.GroupNorm(1, base_dim)
        elif self.input_proj_type == 1:
            self.root_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.root_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.skeleton_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.skeleton_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.extra_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.extra_joint_norm = nn.GroupNorm(gp_norm, base_dim)
        elif self.input_proj_type == 2:
            self.root_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.root_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.skeleton_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.skeleton_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.extra_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.extra_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.obj_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.obj_joint_norm = nn.GroupNorm(gp_norm, base_dim)
        elif self.input_proj_type == 3:
            self.root_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.root_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.skeleton_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.skeleton_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            #self.extra_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            #self.extra_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.obj_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.obj_joint_norm = nn.GroupNorm(gp_norm, base_dim)
        elif self.input_proj_type == 4:
            self.root_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.root_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.skeleton_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.skeleton_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.extra_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.extra_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.extra_contact_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.extra_contact_norm = nn.GroupNorm(gp_norm, base_dim)
            self.obj_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.obj_joint_norm = nn.GroupNorm(gp_norm, base_dim)
        elif self.input_proj_type == 5:
            self.root_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.root_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.skeleton_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.skeleton_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.extra_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.extra_joint_norm = nn.GroupNorm(gp_norm, base_dim)
            self.extra_contact_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.extra_contact_norm = nn.GroupNorm(gp_norm, base_dim)
            self.obj_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
            self.obj_joint_norm = nn.GroupNorm(gp_norm, base_dim)
        else:
            pass

        # 64, 128, 256
        modules = []
        out_ch = base_dim
        up_layers = [int(x) for x in up_layers.split(",")]
        for i in range(num_layers):
            in_ch = out_ch
            if i + 1 in up_layers:
                out_ch = in_ch * 2
            modules.append(GCNBlock(in_ch, out_ch, A.clone(), residual=True,
                                        time_embed_dim=time_embed_dim, text_emb_dim=text_feat_dim,
                                        gcn_group_size=gp_norm * (out_ch // base_dim) if gp_norm != 0 else 1,
                                        tcn_group_size=1, pos_emb=pos_emb, joint_num=self.graph.num_node))

        self.blocks = nn.ModuleList(modules)
        self.output_proj = nn.Conv2d(out_ch, joint_dim, 1, 1, 0)

        self.time_proj = nn.Linear(time_embed_dim, text_feat_dim)

        #使用Pct的维度是1024，使用pointnet是512
        self.obj_proj = nn.Linear(1024, text_feat_dim)

    def proj_and_norm_input(self, x):
        if self.input_proj_type == 0:
            ret = self.input_norm(self.input_proj(x))

            if torch.any(torch.isnan(ret)) and ret.device.index == 0:
                tmp = self.input_proj(x)
                print("after input proj and norm", ret.shape, ret.device)
                print(x.mean(), x.max(), x.min(), x.std())
                print(tmp.mean(), tmp.max(), tmp.min(), tmp.std())
                print(self.input_norm.weight)
                print(self.input_norm.bias)
                print("="*20, "linear", "="*20)
                print(self.input_proj.weight[0])
                print(self.input_proj.bias)
                exit()
            elif torch.any(torch.isnan(ret)):
                exit()
        elif self.input_proj_type == 1:
            root_joint = self.root_joint_norm(self.root_joint_proj(x[..., :1]))
            skeleton_joint = self.skeleton_joint_norm(self.skeleton_joint_proj(x[..., 1:-1]))
            extra_joint = self.extra_joint_norm(self.extra_joint_proj(x[..., -1:]))
            ret = torch.cat([root_joint, skeleton_joint, extra_joint], dim=-1)
        elif self.input_proj_type == 2:
            root_joint = self.root_joint_norm(self.root_joint_proj(x[..., :1]))
            skeleton_joint = self.skeleton_joint_norm(self.skeleton_joint_proj(x[..., 1:-2]))
            extra_joint = self.extra_joint_norm(self.extra_joint_proj(x[..., -2:-1]))
            obj_joint = self.obj_joint_norm(self.obj_joint_proj(x[..., -1:]))
            ret = torch.cat([root_joint, skeleton_joint, extra_joint,obj_joint], dim=-1)
        elif self.input_proj_type == 3:
            root_joint = self.root_joint_norm(self.root_joint_proj(x[..., :1]))
            skeleton_joint = self.skeleton_joint_norm(self.skeleton_joint_proj(x[..., 1:-1]))
            obj_joint = self.obj_joint_norm(self.obj_joint_proj(x[..., -1:]))
            ret = torch.cat([root_joint, skeleton_joint,obj_joint], dim=-1)
        elif self.input_proj_type == 4:
            # import pdb
            # pdb.set_trace()
            root_joint = self.root_joint_norm(self.root_joint_proj(x[..., :1]))
            skeleton_joint = self.skeleton_joint_norm(self.skeleton_joint_proj(x[..., 1:-3]))
            extra_joint = self.extra_joint_norm(self.extra_joint_proj(x[..., -3:-2]))
            extra_contact = self.extra_contact_norm(self.extra_contact_proj(x[..., -2:-1]))
            obj_joint = self.obj_joint_norm(self.obj_joint_proj(x[..., -1:]))
            ret = torch.cat([root_joint, skeleton_joint, extra_joint,extra_contact,obj_joint], dim=-1)
        else:
            pass
        return ret

    def forward(self, x, x_mask, timestep, text,obj_emb):
        
        if torch.any(torch.isnan(x)):
            print("input x", x.shape, x.device)
        x = rearrange(x, "B L V C -> B C L V").contiguous()

        x = self.proj_and_norm_input(x)
        if torch.any(torch.isnan(x)):
            print("after input proj and norm", x.shape, x.device)

        timestep_emb = timestep_embedding(timestep, self.time_embed_dim)



        text_time_emb = rearrange(self.time_proj(timestep_emb), "B C -> B 1 C")
        if obj_emb is not None:
            
            obj_emb=self.obj_proj(obj_emb)
            text_feat = torch.cat([text_time_emb, text["hidden"],obj_emb], dim=1)
            if text["mask"].shape[1]==1:
                text_mask = torch.cat([torch.ones([x.shape[0], 1], device=x.device, dtype=torch.bool),
                                   text["mask"],
                                   torch.zeros(obj_emb.shape[0],obj_emb.shape[1],device=x.device, dtype=torch.bool)], dim=1)
            else:
                text_mask = torch.cat([torch.ones([x.shape[0], 1], device=x.device, dtype=torch.bool),
                                   text["mask"],
                                   torch.ones(obj_emb.shape[0],obj_emb.shape[1],device=x.device, dtype=torch.bool)], dim=1)
        else:
            text_feat = torch.cat([text_time_emb, text["hidden"]], dim=1)
            text_mask = torch.cat([#torch.ones(obj_emb.shape[0],obj_emb.shape[1],device=x.device, dtype=torch.bool),
                                torch.ones([x.shape[0], 1], device=x.device, dtype=torch.bool),
                                text["mask"]], dim=1)
        
        for i, block in enumerate(self.blocks):
            
            x = block(x, x_mask, text_feat, text_mask, timestep_emb)

        x = self.output_proj(x)
        x = rearrange(x, "B C L V -> B L V C")
        return x


class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, residual=False,
                 time_embed_dim=256, text_emb_dim=512, joint_num=23, pointnet_emb=256,pos_emb=None, **kwargs):
        super().__init__()
        
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'mstcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_ch, out_ch, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_ch, out_ch, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_ch, out_ch, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_ch == out_ch) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_ch, out_ch, kernel_size=1, stride=stride)

        if pos_emb is None:
            self.pos_emb = None
        elif pos_emb == "cos":
            self.pos_emb = PositionEmbedding(196, out_ch, dropout=0.1, grad=False)
        elif pos_emb == "learn":
            self.pos_emb = PositionEmbedding(196, out_ch, dropout=0.1, grad=True, randn_norm=True)
        else:
            raise ValueError(f"pose emb {pos_emb} not implemented!")

        self.time_emb_proj = nn.Linear(time_embed_dim, in_ch)
        self.proj_before_cross_att = nn.Linear((joint_num-1) * out_ch, out_ch)
        self.proj_before_cross_att2 = nn.Linear(1 * out_ch, out_ch)

        self.proj_after_cross_att = nn.Linear(out_ch, (joint_num-1) * out_ch)
        self.proj_after_cross_att2 = nn.Linear(out_ch, 1 * out_ch)


        cross_att = nn.TransformerDecoderLayer(d_model=out_ch, nhead=1, dim_feedforward=out_ch * 2,
                                                    batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(cross_att, num_layers=4)
        cross_att2 = nn.TransformerDecoderLayer(d_model=out_ch, nhead=1, dim_feedforward=out_ch * 2,
                                                    batch_first=True)
        self.transformer_decoder2 = nn.TransformerDecoder(cross_att2, num_layers=4)


        self.text_proj = nn.Linear(text_emb_dim, out_ch)
        # self.layer_norm = nn.LayerNorm()

    def forward(self, x, x_mask, text, text_mask, timestep,A=None):
        """Defines the computation performed at every call."""
        if torch.any(torch.isnan(x)):
            print("before gcn", x.shape)
        
        res = self.residual(x)
        x = x + rearrange(self.time_emb_proj(timestep), 'b c -> b c 1 1')
        x = self.relu(self.tcn(self.gcn(x, A)) + res)

        if torch.any(torch.isnan(x)):
            print("after gcn", x.shape)


        # reshape and cross attention
        x_human=x[...,:-1]
        x_obj=x[...,-1:]
        res_human = x_human.clone()
        res_obj = x_obj.clone()
        x_human = rearrange(x_human, "n c t v -> n t (c v)")
        x_obj = rearrange(x_obj, "n c t v -> n t (c v)")
        x_human = self.proj_before_cross_att(x_human)
        x_obj = self.proj_before_cross_att2(x_obj)

        if self.pos_emb is not None:
            #x = x + self.pos_emb(x)
            x_human = self.pos_emb(x_human)
            x_obj = self.pos_emb(x_obj)
        if torch.any(torch.isnan(x)):
            print("before attention", res.shape)

        
        x_cur_human = self.transformer_decoder(tgt=x_human, tgt_key_padding_mask=~x_mask,
                               memory=self.text_proj(text), memory_key_padding_mask=~text_mask)
        x_cur_obj = self.transformer_decoder2(tgt=x_obj, tgt_key_padding_mask=~x_mask,
                               memory=self.text_proj(text), memory_key_padding_mask=~text_mask)

        x_cur_human = self.proj_after_cross_att(x_cur_human)
        x_cur_obj = self.proj_after_cross_att2(x_cur_obj)
        x_cur_human = rearrange(x_cur_human, "n t (c v) -> n c t v", c=res_human.shape[1])
        x_cur_obj = rearrange(x_cur_obj, "n t (c v) -> n c t v", c=res_obj.shape[1])
        x_cur_human=x_cur_human+res_human
        x_cur_obj=x_cur_obj+res_obj
        x=torch.cat((x_cur_human,x_cur_obj),dim=-1)

        if torch.any(torch.isnan(x)):
            print("after attention", x.shape)
            exit()
        return x





##将文本特征和物体点云特征分开，再add
class ChainHOI(nn.Module):
    def __init__(
            self,
            layout="hml3d",
            joint_dim=12,
            obj_clound_dim: int =256,
            text_feat_dim: int=512,
            time_embed_dim: int=512,
            model_dim: int=256,
            trans_layers: int=6,
            trans_layers2: int=4,
            obj_trans_layer: int = 4,
            num_layers: int = 6,
            base_dim: int = 64,
            up_layers: str = "2,4",
            gp_norm: int = 4,
            pos_emb: str = 'cos',
            arch: int = 1,
            input_proj_type: int = 0,
            num_tokens: int = 32,
            has_obj: bool=False,
            cond_type: str="add",
            two_token: bool=False,
            connect_all: bool=False,
            **kwargs
    ):
        super().__init__()
        self.arch = arch

        self.time_embed_dim = time_embed_dim
        self.text_feat_dim = text_feat_dim
        self.num_layers = num_layers
        self.input_proj_type = input_proj_type
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.num_tokens = num_tokens
        
        self.time_emb = TimestepEmbedding(time_embed_dim, time_embed_dim)
        self.has_obj=has_obj
        extra_joint = self.arch == 1
        self.graph = Graph(layout=layout, mode='spatial', extra_joint=extra_joint)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)



        
        self.root_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
        self.root_joint_norm = nn.GroupNorm(gp_norm, base_dim)
        self.skeleton_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
        self.skeleton_joint_norm = nn.GroupNorm(gp_norm, base_dim)
        self.extra_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
        self.extra_joint_norm = nn.GroupNorm(gp_norm, base_dim)
        self.obj_joint_proj = nn.Conv2d(joint_dim, base_dim, 1, 1, 0)
        self.obj_joint_norm = nn.GroupNorm(gp_norm, base_dim)


        modules = []
        out_ch = base_dim
        up_layers = [int(x) for x in up_layers.split(",")]
        for i in range(num_layers):
            in_ch = out_ch
            if i + 1 in up_layers:
                out_ch = in_ch * 2

            block = BaseBlock(
                in_ch, out_ch,model_dim,trans_layers,trans_layers2, A.clone(),
                text_dim=text_feat_dim, obj_dim=obj_clound_dim,
                group_size=gp_norm * (out_ch // base_dim) if gp_norm != 0 else 1,
                arch=kwargs["block_arch"], temp_arch=kwargs["temp_arch"],
                reduce_joint=kwargs["reduce_joint"], joint_num=self.graph.num_node,
                res=kwargs.get("res", False), down_sample=kwargs.get("down_sample", 8),
                layout=layout, joint_arch=kwargs.get("joint_arch", "sa"),
                nheads=kwargs.get("nheads", 1), use_rpe=kwargs.get("rpe", False),
                post_proj=kwargs.get("post_proj", None),pos_emb=pos_emb,cond_type=cond_type,
                two_token=two_token,connect_all=connect_all
            )
            modules.append(block)

        self.blocks = nn.ModuleList(modules)
        self.output_proj = nn.Conv2d(out_ch, joint_dim, 1, 1, 0)


        #condition处理
        self.time_proj = nn.Linear(time_embed_dim, text_feat_dim)
        self.text_proj = nn.Linear(text_feat_dim, text_feat_dim)
        self.obj_proj = nn.Linear(obj_clound_dim, text_feat_dim)


    def proj_and_norm_input(self, x):
        
        root_joint = self.root_joint_norm(self.root_joint_proj(x[..., :1]))
        skeleton_joint = self.skeleton_joint_norm(self.skeleton_joint_proj(x[..., 1:-2]))
        extra_joint = self.extra_joint_norm(self.extra_joint_proj(x[..., -2:-1]))
        obj_joint = self.obj_joint_norm(self.obj_joint_proj(x[..., -1:]))
        ret = torch.cat([root_joint, skeleton_joint, extra_joint,obj_joint], dim=-1)
        return ret

    def forward(self, x, x_mask, timestep, text,obj_emb):
        
        x = rearrange(x, "B L J C -> B C L J").contiguous()
        x = self.proj_and_norm_input(x)

        timestep_emb = self.time_proj(timestep_embedding(timestep, self.time_embed_dim))
        #obj_emb = self.obj_proj(obj_emb)
        #obj_emb = self.alpha * obj_emb
        text_feat = text["hidden"]
        text_feat1 = torch.cat([timestep_emb.unsqueeze(dim=1),text_feat], dim=1)
        text_mask1 = torch.cat([torch.ones([x.shape[0], 1], device=x.device, dtype=torch.bool),
                               text["mask"]], dim=1)
        text_feat2 = torch.cat([timestep_emb.unsqueeze(dim=1),obj_emb], dim=1)
        text_mask2 = torch.cat([torch.ones([x.shape[0], 1], device=x.device, dtype=torch.bool),
                        torch.ones(obj_emb.shape[0],obj_emb.shape[1],device=x.device, dtype=torch.bool)], dim=1)
        
        
        for i, block in enumerate(self.blocks):
            x = block(x, x_mask, text_feat1, text_mask1,text_feat2,text_mask2)

        x = self.output_proj(x)
        x = rearrange(x, "B C L J -> B L J C")
        return x


class STGCNBaseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, group_size=1):
        super().__init__()
        self.gcn = unit_gcn(in_ch, out_ch, A, group_size=group_size)
        self.tcn = mstcn(out_ch, out_ch, group_size=1)
        self.relu = nn.ReLU()

        if in_ch == out_ch:
            self.residual = lambda x: x
        else:
            # todo: 是否是直接一个1x1卷积就行了？
            self.residual = unit_tcn(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.tcn(self.gcn(x)) + res)
        return x

class TempResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, arch, text_dim, obj_dim,time_dim,
                 nheads=1, norm_pos=None, gp_norm=1,has_obj=False,
                 reduce_joint="avg", joint_num=22):
        super().__init__()
        self.arch = arch
        self.reduce_joint = reduce_joint

        self.norm_pos = norm_pos
        if self.norm_pos is not None:
            self.norm = nn.GroupNorm(gp_norm, in_ch if self.norm_pos=="pre" else out_ch)

        if self.reduce_joint == "fc":
            self.proj_pre = nn.Linear(in_ch*joint_num, out_ch)
            self.proj_post = nn.Linear(out_ch, out_ch*joint_num)
            # self.proj_pre = nn.Sequential(nn.Linear(in_ch * joint_num, out_ch), nn.SiLU())
            # self.proj_post = nn.Sequential(nn.Linear(out_ch, out_ch * joint_num), nn.SiLU())
        elif self.reduce_joint == "avg":
            self.input_proj = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.SiLU()
            )
        else:
            raise ValueError(f"{self.reduce_joint} not supported!")

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, out_ch),
            nn.SiLU()
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, out_ch),
            nn.SiLU()
        )
        if has_obj is True:
            self.obj_proj = nn.Sequential(
                nn.Linear(obj_dim, out_ch),
                nn.SiLU()
            )
        self.pos_emb = PositionEmbedding(256, out_ch, dropout=0.1) # nn.Identity() #


        if self.arch == 'cross_attn':
            self.cross_att = nn.TransformerDecoderLayer(out_ch, nheads, dim_feedforward=out_ch*2, batch_first=True)
            # self.mha = nn.MultiheadAttention(out_ch, nheads, dropout=0.1, batch_first=True)
        elif self.arch == 'self_attn':
            self.mha = nn.TransformerEncoderLayer(out_ch, nheads, dim_feedforward=out_ch*2, batch_first=True)
        else:
            raise ValueError(f"{self.arch} not supported!")

    def forward(self, x, x_mask, text_emb, time_emb,obj_emb):
        if self.norm_pos == "pre":
            x = self.norm(x)
        J = x.shape[-1]
        if self.reduce_joint == "avg":
            x = rearrange(x, "B C L V -> B L V C")
            x = self.input_proj(x)
            x = reduce(x, "B L V C -> B L C", reduction="mean")
        elif self.reduce_joint == "fc":
            x = rearrange(x, "B C L V -> B L (C V)")
            x = self.proj_pre(x)
        
        new_text_emb = self.text_proj(text_emb)
        new_time_emb = self.time_proj(time_emb)
        if obj_emb is not None:
            new_obj_emb=self.obj_proj(obj_emb)
            cond = torch.stack([new_text_emb,new_obj_emb, new_time_emb], dim=1)
        else:
            cond = torch.stack([new_text_emb, new_time_emb], dim=1)
        if self.arch == "cross_attn":
            # todo: 是否需要加上time idx emb
            x = self.pos_emb(x)
            x = self.cross_att(tgt=x, tgt_key_padding_mask=~x_mask, memory=cond)
            # x = self.mha(x, cond, cond)
        elif self.arch == "self_attn":
            x_att = torch.cat([cond, x], dim=1)
            x_att = self.pos_emb(x_att)
            pad_mask = torch.cat([torch.ones([x.shape[0], 2], device=x.device, dtype=torch.bool), x_mask], dim=1)
            x_att =  self.mha(src=x_att, src_key_padding_mask=~pad_mask)
            x = x_att[:, 2:]

        if self.reduce_joint == "avg":
            x = repeat(x, "B L C -> B C L J", J=J)
        elif self.reduce_joint == "fc":
            x = self.proj_post(x)
            x = rearrange(x, "N T (C J) -> N C T J", J=J)
        
        if self.norm_pos == "post":
            x = self.norm(x)
        return x




class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0, q_dim=None, k_dim=None, v_dim=None, qkv_proj=False,
                 rpe=False, rpe_size=0):
        super().__init__()
        assert dim % heads == 0
        head_dim = dim // heads
        self.heads = heads
        if qkv_proj and q_dim is None:
            q_dim, k_dim, v_dim = dim, dim, dim
        self.q_proj = nn.Linear(q_dim, dim) if q_dim is not None else nn.Identity()
        self.k_proj = nn.Linear(k_dim, dim) if k_dim is not None else nn.Identity()
        self.v_proj = nn.Linear(v_dim, dim) if v_dim is not None else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        if rpe:
            self.rpe = nn.Parameter(torch.zeros(heads, 2 * rpe_size - 1))
            self.register_buffer("rpe_idx", self.build_rpe_idx(rpe_size))
        else:
            self.rpe = None

    def forward(self, q, k, v, q_padding_mask=None, k_padding_mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q, k, v = map(lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.heads), [q, k, v])
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if self.rpe is not None:
            sub_idx = self.rpe_idx[:q.shape[1], :q.shape[1]]
            rpe = self.rpe[:, sub_idx.view(-1)].view([-1, sub_idx.shape[0], sub_idx.shape[1]])
            dots = dots + rpe
        if k_padding_mask is not None:
            mask = repeat(k_padding_mask, "b lk -> b h lq lk", h=self.heads, lq=q.shape[2])
            if dots.dtype == torch.float16:
                dots.masked_fill_(mask, -65504)
            else:
                dots.masked_fill_(mask, -1e9)
        attn = self.softmax(dots)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.proj_out(out)
        if q_padding_mask is not None:
            out.masked_fill_(q_padding_mask.unsqueeze(dim=-1), 0)
        return out

    def build_rpe_idx(self, rpe_size):
        tmp = torch.arange(0, rpe_size, dtype=torch.long)
        rpe = tmp.unsqueeze(dim=0).repeat([rpe_size, 1]) - tmp.unsqueeze(dim=1)
        return rpe + rpe_size - 1




class BaseBlock(nn.Module):
    def __init__(self, in_ch, out_ch,model_dim,trans_layers,trans_layers2, A, text_dim,obj_dim, group_size, arch, nheads, use_rpe,
                 temp_arch, reduce_joint, joint_num, res, down_sample, layout, joint_arch, post_proj,pos_emb,cond_type,two_token,connect_all):
        super().__init__()
        self.res = res
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0),
            nn.ReLU(),
            nn.GroupNorm(1, out_ch)
        )
        self.gcn_block = STGCNBaseBlock(out_ch, out_ch, A, group_size)
        self.temp_block = Semantic_ConBlock(out_ch, temp_arch, text_dim,obj_dim, nheads=nheads,
                                          reduce_joint=reduce_joint, joint_num=joint_num, res=res, post_proj=post_proj,trans_layer=trans_layers2,cond_type=cond_type)
        self.joint_block = KIM(out_ch, model_dim=model_dim, trans_layers=trans_layers,down_sample=down_sample, arch=joint_arch,
                                             layout=layout, nheads=nheads,pos_emb=pos_emb,cond_type=cond_type,two_token=two_token,connect_all=connect_all)

        if arch == 0:#add
            pass
        elif arch == 1:#add2
            self.proj =  nn.Conv2d(out_ch, out_ch, 1, 1, 0)
        elif arch == 2:#cat
            self.proj = nn.Conv2d(out_ch*2, out_ch, 1, 1, 0)
        elif arch == 3:#cat2
            self.proj = nn.Conv2d(out_ch * 2, out_ch, 1, 1, 0)
        elif arch == 4:#norm1
            self.gp = 1
            self.proj = nn.Conv2d(out_ch, out_ch, 1, 1, 0)
            self.predict_norm = nn.Sequential(nn.Linear(out_ch, out_ch), nn.SiLU(), nn.Linear(out_ch, 2*out_ch))
        self.arch = arch
        self.norm = nn.Sequential(nn.GroupNorm(1, out_ch))

    def forward(self, x, x_mask, text_emb1, text_mask1,text_emb2, text_mask2):

        rx = self.input_proj(x)
        x_s = self.gcn_block(rx)
        x_t = self.temp_block(rx, x_mask, text_emb1, text_mask1,text_emb2, text_mask2)
        x_j = self.joint_block(x_s, x_t, x_mask,text_emb1, text_mask1,text_emb2, text_mask2)
        #x_j = self.joint_block(x_s, x_t, x_mask)

        if self.arch == 0:
            x = self.norm(rx + x_j)
        elif self.arch == 1:
            x = self.norm(self.proj(x_s + x_j))
        elif self.arch == 2:
            x = torch.cat([rx, x_j], dim=1)
            x = self.norm(self.proj(x))
        elif self.arch == 3:
            x = torch.cat([x_s, x_j], dim=1)
            x = self.norm(self.proj(x))
        elif self.arch == 4:
            x_s = rearrange(x_s, "B (N C) L V -> B N C L V", N=self.gp)
            x_mean, x_std = torch.mean(x_s, dim=[2, 4], keepdim=True), torch.std(x_s, dim=[2, 4], keepdim=True)
            x_j = reduce(self.proj(x_j), "B C L J -> B L C", "mean")
            scale_shift = self.predict_norm(x_j).chunk(2, dim=-1)
            scale, shift = [repeat(_, "B L (N C) -> B N C L A", A=1, N=self.gp) for _ in scale_shift]
            x = (x_s - x_mean) / x_std
            x = (1 + scale) * x + shift
            x = rearrange(x, "B N C L V -> B (N C) L V")

        if self.res:
            x = x + rx

        return x





class Semantic_ConBlock(nn.Module):
    def __init__(self, out_ch, arch, text_dim,obj_dim,
                 nheads=1, reduce_joint="conv1", joint_num=22, use_rpe=False, res=False, post_proj=False,trans_layer=4,cond_type="add"):
        super().__init__()
        self.arch = arch
        self.reduce_joint = reduce_joint
        self.use_post_proj = post_proj
        self.res = res
        self.norm = nn.LayerNorm(out_ch)
        self.cond_type = cond_type
        group_size = 1

        if self.reduce_joint == "conv1":
            self.input_proj = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 1, 1, 0),
                # nn.SiLU()
                # nn.ReLU(),
                # nn.GroupNorm(group_size, out_ch),
            )
        elif self.reduce_joint == "conv2":
            self.input_proj = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, (1, joint_num), (1, joint_num), 0, groups=out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 1, 1, 0),
                # nn.SiLU(),
                # nn.GroupNorm(group_size, out_ch)
            )
        elif self.reduce_joint == "conv3":
            self.input_proj = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 1, 1, 0),
                nn.SiLU(),
                nn.GroupNorm(1, out_ch),
                nn.Conv2d(out_ch, out_ch, (1, joint_num), (1, joint_num), 0, groups=out_ch),
                # nn.SiLU(),
                # nn.GroupNorm(1, out_ch),
            )
        elif self.reduce_joint == "conv4":
            self.input_proj = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, (1, joint_num), (1, joint_num), 0),
                # nn.SiLU()
                # nn.ReLU(),
                # nn.GroupNorm(group_size, out_ch),
            )
            # self.input_proj2 = nn.Linear(24*out_ch, out_ch)
            # self.out_proj2 = nn.Linear(24*out_ch, out_ch)
        else:
            raise ValueError(f"{self.reduce_joint} not supported!")

        self.text_proj = nn.Sequential(nn.Linear(text_dim, out_ch),nn.SiLU())
        self.obj_proj = nn.Sequential(nn.Linear(obj_dim, out_ch),nn.SiLU())

        self.pos_emb = nn.Identity() if use_rpe else PositionEmbedding(256, out_ch, dropout=0.1)
        self.pos_emb2 = nn.Identity() if use_rpe else PositionEmbedding(256, out_ch, dropout=0.1)


        if self.arch in ['SA', 'CA']:
            self.mha = MultiHeadAttention(out_ch, nheads, dropout=0.1, rpe=use_rpe, rpe_size=200)
        elif self.arch == "SACA":
            # self.mha1 = MultiHeadAttention(out_ch, nheads, dropout=0.1, qkv_proj=True)
            self.mha1 = MultiHeadAttention(out_ch, nheads, dropout=0.1, qkv_proj=True, rpe=use_rpe, rpe_size=200)
            self.norm1 = nn.LayerNorm(out_ch)
            self.mha2 = MultiHeadAttention(out_ch, nheads, dropout=0.1, q_dim=out_ch)
        elif self.arch in ["cat-SA", "cat-CA"]:
            self.mha = MultiHeadAttention(out_ch, nheads, dropout=0.1, rpe=use_rpe, rpe_size=200,
                                          q_dim=out_ch*2, k_dim=out_ch*2, v_dim=out_ch*2)
        elif self.arch == "OCA":
            cross_att = nn.TransformerDecoderLayer(d_model=out_ch, nhead=nheads, dim_feedforward=out_ch*2, batch_first=True)
            self.decoder = nn.TransformerDecoder(cross_att, num_layers=trans_layer)
            cross_att2 = nn.TransformerDecoderLayer(d_model=out_ch, nhead=nheads, dim_feedforward=out_ch*2, batch_first=True)
            self.decoder2 = nn.TransformerDecoder(cross_att2, num_layers=trans_layer)
            if cond_type=="concat":
                self.after_proj = nn.Sequential(nn.Linear(2 * out_ch, out_ch),nn.SiLU())
        else:
            raise ValueError(f"{self.arch} not supported!")
        if self.use_post_proj:
            self.post_proj = nn.Sequential(nn.Linear(out_ch, out_ch), nn.SiLU())

    def forward(self, x, x_mask, text_emb1, text_mask1,text_emb2, text_mask2):
        B=x.shape[0]
        x = self.input_proj(x)
        x = reduce(x, "B C L J -> B L C", "mean")
        # x = rearrange(x, "B C L J -> B L (J C)")#第一种
        # x = self.input_proj2(x)#第一种
        
        #x = rearrange(x, "B C L J -> (B J) L C")#第二种
        

        rx = x
        # cond1 = self.text_proj(text_emb1).repeat(24,1,1)
        # cond_mask1 = text_mask1.repeat(24,1)

        # cond2 = self.obj_proj(text_emb2).repeat(24,1,1)
        # cond_mask2 = text_mask2.repeat(24,1)
        cond1 = self.text_proj(text_emb1)
        cond_mask1 = text_mask1

        cond2 = self.obj_proj(text_emb2)
        cond_mask2 = text_mask2

        if self.arch == "CA":
            x = self.pos_emb(x)
            x = self.mha(x, cond, cond, q_padding_mask=~x_mask, k_padding_mask=~cond_mask)
            if self.res:
                x = self.norm(x + rx)
        elif self.arch == "SA":
            x_att = torch.cat([cond, x], dim=1)
            x_att = self.pos_emb(x_att)
            mask = torch.cat([cond_mask, x_mask], dim=1)
            x_att = self.mha(x_att, x_att, x_att, q_padding_mask=~mask, k_padding_mask=~mask)
            x = x_att[:, cond.shape[1]:]
            if self.res:
                x = self.norm(x + rx)
        elif self.arch == "SACA":
            nx = self.pos_emb(x)
            x = self.norm1(x + self.mha1(nx, nx, nx, q_padding_mask=~x_mask, k_padding_mask=~x_mask))
            x = self.norm(x + self.mha2(x, cond, cond, q_padding_mask=~x_mask, k_padding_mask=~cond_mask))
        elif self.arch == "cat-CA":
            pass
            # x = torch.cat([x, new_text_emb.unsqueeze(dim=1)], dim=1)
            # x = self.pos_emb(x)
            # x = self.mha(x, cond, cond, q_padding_mask=~x_mask)
            # if self.res:
            #     x = self.norm(x + rx)
        elif self.arch == "OCA":
            
            x1 = self.pos_emb(x)
            x2 = self.pos_emb2(x)
            x1 = self.decoder(tgt=x1, tgt_key_padding_mask=~x_mask, memory=cond1, memory_key_padding_mask=~cond_mask1)
            x2 = self.decoder2(tgt=x2, tgt_key_padding_mask=~x_mask, memory=cond2, memory_key_padding_mask=~cond_mask2)
            if self.cond_type == "concat":
                x = torch.cat([x1,x2],dim=2)
                x = self.after_proj(x)
            else:
                x = torch.add(x1,x2)

        if self.use_post_proj:
            x = self.post_proj(x)

        # x = rearrange(x,"(B J) L C -> B J L C",B=B)
        # x = rearrange(x,"B J L C -> B L (J C)")
        # x = self.out_proj2(x)
        return x




class KIM(nn.Module):
    def __init__(self, out_ch,model_dim,trans_layers, joint_num=24, down_sample=8, arch=1,layout='behave',
                 nheads=1, use_rpe=False, partition=1,pos_emb="cos",cond_type="add",two_token=False,connect_all=False):
        super().__init__()
        self.arch = arch
        self.num_keys = joint_num
        self.cond_type = cond_type
        self.down_sample = down_sample
        self.input_proj = nn.Sequential(nn.Conv2d(out_ch*2, out_ch, 1, 1, 0))
        self.two_token = two_token
        self.connect_all = connect_all
        
        if self.two_token:
            self.limb_token = nn.Parameter(torch.randn([3, out_ch]))
        else:
            self.limb_token = nn.Parameter(torch.randn([6, out_ch]))
        
        # cross_att = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nheads, dim_feedforward=model_dim*2,activation=F.gelu,
        #                                            batch_first=True,norm_first=False)
        cross_att2 = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nheads, dim_feedforward=model_dim*2,activation=F.gelu,
                                                    batch_first=True,norm_first=False)
        # cross_att3 = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nheads, dim_feedforward=model_dim*2,activation=F.gelu,
        #                                             batch_first=True,norm_first=False)   

        

        cross_att = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nheads, dim_feedforward=model_dim*2,activation=F.gelu,
                                                   batch_first=True,norm_first=False)
        self.transformer_decoder = nn.TransformerDecoder(cross_att, num_layers=trans_layers)
        self.transformer_decoder2 = nn.TransformerDecoder(cross_att2, num_layers=trans_layers)
        #self.transformer_decoder3 = nn.TransformerDecoder(cross_att3, num_layers=trans_layers)
        self.ske_proj = nn.Linear(out_ch, model_dim)
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.text_proj = nn.Linear(512, model_dim)
        self.text_proj2 = nn.Linear(512, model_dim)
        self.x_cur_proj = nn.Linear(out_ch, model_dim)
        self.x_cur_proj2 = nn.Linear(out_ch, model_dim)
        if self.two_token:
            self.proj_after_cross_att = nn.Linear(3 * model_dim, joint_num * out_ch)
        else:
            self.proj_after_cross_att = nn.Linear(6 * model_dim, joint_num * out_ch)
        #self.proj_after_cross_att = nn.Linear(model_dim, out_ch)
        self.m_output_proj = nn.Linear(model_dim, out_ch)
        if self.two_token:
            indices = [
                [0,1,2,4, 5, 7,8, 10,11],
                [0, 3, 6, 9, 12,13,14, 15,16,17,18,19,20,21],
                [0,10,11,12,16,17,20,21,23]
            ]
        else:
            if self.connect_all:
                indices = [
                    [0, 2, 5, 8, 11],
                    [0, 1, 4, 7, 10],
                    [0, 3, 6, 9, 12, 15],
                    [9, 14, 17, 19, 21],
                    [9, 13, 16, 18, 20],
                    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23]
                ]
            else:
                indices = [
                    [0, 2, 5, 8, 11],
                    [0, 1, 4, 7, 10],
                    [0, 3, 6, 9, 12, 15],
                    [9, 14, 17, 19, 21],
                    [9, 13, 16, 18, 20],
                    [0,10,11,12,16,17,20,21,23]
                ]

        # 初始化 mask 矩阵为全零，形状为 [6, 24]
        if self.two_token:
            self.mask = torch.zeros((3, self.num_keys), dtype=torch.float32)
        else:
            self.mask = torch.zeros((6, self.num_keys), dtype=torch.float32)

        # 设置 mask 中与 query 相关的 key 索引为 1
        for i, idx_list in enumerate(indices):
            self.mask[i, idx_list] = 1
        if self.cond_type == "concat":
            self.m_output_proj2 = nn.Linear(2 * model_dim, model_dim)

    


    def forward(self, x_s, x_t, x_mask,text_emb1, text_mask1,text_emb2, text_mask2):
        #x_s:torch.Size([32, 64, 196, 25])
        #x_t:torch.Size([32, 196, 64])

        x = torch.cat([x_s, repeat(x_t, "B L C -> B C L A", A=x_s.shape[-1])], dim=1)#torch.Size([32, 128, 196, 25])
        x = self.input_proj(x)
        
        
        x_ = rearrange(x,"B C L V -> (B L) V C")

        
        x_proj = self.ske_proj(x_)
        limb_query = self.limb_token.unsqueeze(0).repeat(text_emb1.shape[0],1,1)
        x_cur1 = self.transformer_decoder(tgt=self.x_cur_proj(limb_query),memory=self.text_proj(text_emb1),memory_key_padding_mask=~text_mask1)
        x_cur2 = self.transformer_decoder2(tgt=self.x_cur_proj2(limb_query),memory=self.text_proj2(text_emb2),memory_key_padding_mask=~text_mask2)
        if self.cond_type == "concat":
            x_cur = torch.cat([x_cur1,x_cur2],dim=-1)
            x_cur = self.m_output_proj2(x_cur)
        else:
            x_cur = torch.add(x_cur1,x_cur2)
        #x_obj=x_[:,23:,:]


        x_cur = x_cur.unsqueeze(1).repeat(1,x.shape[2],1,1)
        x_cur = rearrange(x_cur,"B L V C -> (B L) V C")
        # query 的数量和 key 的数量
        num_queries = limb_query.shape[1]
        # 索引关系
        #contact_point=[0,10,11,12,16,17,20,21]
        

        mask = self.mask.to(x_cur.device)
        x_cur = self.query_proj(x_cur)

        x_cur = self.transformer_decoder(tgt=x_cur,memory=x_proj,memory_mask=1-mask)#x_cur：[B*T,6,model_dim] x_proj[B*T,24,model_dim]


        x_cur = rearrange(x_cur,"(B L) V C -> B L (V C)",B=x.shape[0])
        x_cur = rearrange(self.proj_after_cross_att(x_cur),"B L (V C) -> B L V C",V=self.num_keys)
        x_proj = rearrange(self.m_output_proj(x_proj)," (B L) V C -> B L V C",B=x.shape[0])
        x_proj =torch.add(x_proj,x_cur)
        

        x = rearrange(x_proj,"B L V C -> B C L V")
        #x = torch.cat((x_human,x[...,22:]),dim=-1)
        return x




def joint_partition(joints, layout, partition=1, avg=False):
    ret = []
    assert partition in [1, 2]
    if layout.startswith("hml3d"):
        if partition == 1:
            limbs = [
                [1, 4, 7, 10],
                [2, 5, 8, 11],
                [0, 3, 6, 9, 12, 15],
                [13, 16, 18, 20],
                [14, 17, 19, 21],
                [7, 8, 10, 11, 22]
            ]
        elif partition == 2:
            limbs = [
                [1, 4, 7, 10, 22],
                [2, 5, 8, 11, 22],
                [0, 3, 6, 9, 12, 15],
                [13, 16, 18, 20],
                [14, 17, 19, 21],
            ]
        joint_cnt = torch.ones(23, device=joints.device)
    elif layout.startswith("kit"):
        if partition == 1:
            limbs = [
                [0, 1, 2, 3, 4],
                [5, 6, 7],
                [8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [14, 15, 19, 20, 21]
            ]
        elif partition == 2:
            limbs = [
                [0, 1, 2, 3, 4],
                [5, 6, 7],
                [8, 9, 10],
                [11, 12, 13, 14, 15, 21],
                [16, 17, 18, 19, 20, 21],
            ]
        joint_cnt = torch.ones(22, device=joints.device)
    else:
        raise ValueError(f"{layout} not supported!")

    joint_cnt[limbs[-1]] = joint_cnt[limbs[-1]] + 1
    for limb in limbs:
        if avg:
            ret.append(joints[..., limb].mean(dim=-1)) # B C L
        else:
            ret.append(joints[..., limb].contiguous()) # B C L T
    return ret, limbs, joint_cnt

def get_partition_matrix(layout, partition=1):
    if layout.startswith("hml3d"):
        if partition == 1:
            limbs = [
                [1, 4, 7, 10],
                [2, 5, 8, 11],
                [0, 3, 6, 9, 12, 15],
                [13, 16, 18, 20],
                [14, 17, 19, 21],
                [7, 8, 10, 11, 22]
            ]
        elif partition == 2:
            limbs = [
                [1, 4, 7, 10, 22],
                [2, 5, 8, 11, 22],
                [0, 3, 6, 9, 12, 15],
                [13, 16, 18, 20],
                [14, 17, 19, 21],
            ]
    elif layout.startswith("kit"):
        if partition == 1:
            limbs = [
                [0, 1, 2, 3, 4],
                [5, 6, 7],
                [8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [14, 15, 19, 20, 21]
            ]
        elif partition == 2:
            limbs = [
                [0, 1, 2, 3, 4],
                [5, 6, 7],
                [8, 9, 10],
                [11, 12, 13, 14, 15, 21],
                [16, 17, 18, 19, 20, 21],
            ]
    elif layout.startswith("behave"):
        limbs = [
                [0, 2, 5, 8, 11],
                [0, 1, 4, 7, 10],
                [0, 3, 6, 9, 12, 15],
                [9, 14, 17, 19, 21],
                [9, 13, 16, 18, 20],
            ]
    M = torch.zeros([len(limbs), 22 if layout.startswith("behave") else 22])
    for i, limb in enumerate(limbs):
        M[i, [limb]] = M[i, [limb]] + 1

    return M


import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first

        # Multi-head attention layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation function
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ):
        """
        Forward pass for the custom decoder layer.
        Returns both the output and the cross-attention weights.
        """
        x = tgt

        if self.norm_first:
            # Self-attention block
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)

            # Cross-attention block
            x, cross_attn_weights = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )

            # Feedforward block
            x = x + self._ff_block(self.norm3(x))
        else:
            # Self-attention block
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))

            # Cross-attention block
            x, cross_attn_weights = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask
            )

            # Feedforward block
            x = self.norm3(x + self._ff_block(x))

        return x, cross_attn_weights  # 返回 cross-attention 的权重

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """
        Self-attention block
        """
        output = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return self.dropout1(output)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        """
        Cross-attention block
        """
        output, attn_weights = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True
        )
        return self.dropout2(output), attn_weights

    def _ff_block(self, x):
        """
        Feedforward block
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

