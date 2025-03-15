import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from src.models.utils.embedding import timestep_embedding, TimestepEmbedding, PositionEmbedding, PartialPositionEmbedding
from src.models.utils.utils import lengths_to_mask

class KFG1(nn.Module):
    def __init__(
        self,
        motion_dim=263,
        num_keyframes=6,
        model_dim=512,
        text_feat_dim=512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 8,
        arch: int = 1,
        head_type=1,
        head_drop=0,
        text_type=1,
        pose_only=False,
    ):
        super().__init__()
        self.arch = arch
        self.head_type = head_type
        self.text_type = text_type
        self.pose_only = pose_only

        if type(num_layers) is str:
            enc_layer, dec_layer = [int(x) for x in num_layers.split('-')]
        else:
            enc_layer, dec_layer = num_layers, num_layers

        self.text_proj = nn.Linear(text_feat_dim, model_dim)
        self.m_pos_emb = PartialPositionEmbedding(512, model_dim, 0, grad=False)

        if head_type == 1 or self.pose_only:
            head_input = model_dim
        else:
            head_input = model_dim // 2
        self.pose_head = nn.Sequential(
            nn.Dropout(head_drop),
            # nn.Linear(head_input, head_input),
            # nn.ReLU(True),
            nn.Linear(head_input, motion_dim, bias=False)
        )
        if not self.pose_only:
            self.tpos_head = nn.Sequential(
                nn.Dropout(head_drop),
                # nn.Linear(head_input, head_input),
                # nn.ReLU(True),
                nn.Linear(head_input, 1, bias=False)
            )

        if self.arch in (1, 2, 3):
            self.keyframe_token = nn.Parameter(torch.randn(num_keyframes, model_dim), requires_grad=True)
        else:
            pass

        if self.arch == 1: # encoder only
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    model_dim, num_heads, dim_feedforward,
                    dropout, activation=F.gelu, batch_first=True,
                ), num_layers=enc_layer
            )
        elif self.arch == 2:    # decoder only
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward,
                                           dropout, activation=F.gelu, batch_first=True),
                num_layers=dec_layer
            )
        elif self.arch == 3:    # encoder-decoder
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    model_dim, num_heads, dim_feedforward,
                    dropout, activation=F.gelu, batch_first=True,
                ), num_layers=enc_layer
            )
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward,
                                           dropout, activation=F.gelu, batch_first=True),
                num_layers=dec_layer
            )


    def forward2(self, text, keyframe_idx):
        text_feat = self.text_proj(text['text_emb'])
        B = text_feat.shape[0]

        if self.arch == 1:
            all_tokens = torch.cat([text_feat.unsqueeze(dim=1), self.keyframe_token.repeat([B, 1, 1])], dim=1)
            all_tokens = self.m_pos_emb(all_tokens,
                                        torch.cat([torch.zeros([B, 1], dtype=bool, device=all_tokens.device),
                                                   keyframe_idx + 1],
                                                  dim=1))
            all_tokens = self.encoder(src=all_tokens)
            output = all_tokens[:, 1:]
        elif self.arch == 2:
            output = self.decoder(tgt=self.keyframe_token.repeat([B, 1, 1]),
                                  memory=text_feat, memory_key_padding_mask=~text_mask)
        elif self.arch == 3:
            hidden_states = self.encoder(src=text_feat, src_key_padding_mask=~text_mask)
            output = self.decoder(tgt=self.keyframe_token.repeat([B, 1, 1]),
                                  memory=hidden_states, memory_key_padding_mask=~text_mask)

        if self.pose_only:
            pose = self.pose_head(output)
            time = None
        else:
            if self.head_type == 1:
                pose = self.pose_head(output)
                time = self.tpos_head(output)
            else:
                pose = self.pose_head(output[..., :output.shape[-1] // 2])
                time = self.tpos_head(output[..., output.shape[-1] // 2:])
        return pose, time

    def forward(self, text, keyframe_idx):
        if self.text_type == 2:
            return self.forward2(text, keyframe_idx)

        text_feat = self.text_proj(text['hidden'])
        text_mask = text['mask']
        B = text_feat.shape[0]

        if self.arch == 1:
            all_tokens = torch.cat([self.keyframe_token.repeat([B, 1, 1]), text_feat], dim=1)
            all_tokens = self.m_pos_emb(all_tokens,
                           torch.cat([torch.zeros([B, 1], dtype=bool, device=all_tokens.device), keyframe_idx + 1],
                                     dim=1))
            all_masks = torch.cat([torch.ones([B, self.keyframe_token.shape[0]], dtype=bool,
                                              device=text_feat.device), text_mask], dim=1)
            all_tokens = self.encoder(src=all_tokens, src_key_padding_mask=~all_masks)
            output = all_tokens[:, :self.keyframe_token.shape[0]]
        elif self.arch == 2:
            output = self.decoder(tgt=self.keyframe_token.repeat([B, 1, 1]),
                         memory=text_feat, memory_key_padding_mask=~text_mask)
        elif self.arch == 3:
            hidden_states = self.encoder(src=text_feat, src_key_padding_mask=~text_mask)
            output = self.decoder(tgt=self.keyframe_token.repeat([B, 1, 1]),
                                     memory=hidden_states, memory_key_padding_mask=~text_mask)

        if self.pose_only:
            pose = self.pose_head(output)
            time = None
        else:
            if self.head_type == 1:
                pose = self.pose_head(output)
                time = self.tpos_head(output)
            else:
                pose = self.pose_head(output[..., :output.shape[-1] // 2])
                time = self.tpos_head(output[..., output.shape[-1] // 2:])

        return pose, time


class KFG2(nn.Module):
    def __init__(
        self,
        motion_dim=263,
        num_keyframes=6,
        model_dim=512,
        text_feat_dim=512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 8,
        arch: int = 1,
        head_type=1,
        head_drop=0,
        text_type=1,
        pose_only=False,
        max_motion_length=384,
    ):
        super().__init__()
        self.arch = arch
        self.head_type = head_type
        self.text_type = text_type
        self.pose_only = pose_only

        if type(num_layers) is str:
            enc_layer, dec_layer = [int(x) for x in num_layers.split('-')]
        else:
            enc_layer, dec_layer = num_layers, num_layers

        self.m_pos_emb = PartialPositionEmbedding(max_motion_length, model_dim, 0, grad=False, randn_norm=False)

        self.text_proj = nn.Linear(text_feat_dim, model_dim)


        if head_type == 1 or self.pose_only:
            head_input = model_dim
        else:
            head_input = model_dim // 2
        self.pose_head = nn.Sequential(
            nn.Dropout(head_drop),
            # nn.Linear(head_input, head_input),
            # nn.ReLU(True),
            nn.Linear(head_input, motion_dim, bias=False)
        )
        if not self.pose_only:
            self.tpos_head = nn.Sequential(
                nn.Dropout(head_drop),
                # nn.Linear(head_input, head_input),
                # nn.ReLU(True),
                nn.Linear(head_input, 1, bias=False)
            )

        if self.arch in (1, 2, 3):
            self.keyframe_token = nn.Parameter(torch.randn(1, model_dim), requires_grad=True)
        else:
            pass

        if self.arch == 1: # encoder only
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    model_dim, num_heads, dim_feedforward,
                    dropout, activation=F.gelu, batch_first=True,
                ), num_layers=enc_layer
            )
        elif self.arch == 2:    # decoder only
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward,
                                           dropout, activation=F.gelu, batch_first=True),
                num_layers=dec_layer
            )
        elif self.arch == 3:    # encoder-decoder
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    model_dim, num_heads, dim_feedforward,
                    dropout, activation=F.gelu, batch_first=True,
                ), num_layers=enc_layer
            )
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward,
                                           dropout, activation=F.gelu, batch_first=True),
                num_layers=dec_layer
            )


    def forward2(self, text, keyframe_idx):
        text_feat = self.text_proj(text['text_emb'])
        B, L = text_feat.shape[0], text_feat.shape[1]

        if self.arch == 1:
            all_tokens = torch.cat([text_feat.unsqueeze(dim=1),
                                    self.keyframe_token.repeat([B, keyframe_idx.shape[1], 1]), ], dim=1)
            all_tokens = self.m_pos_emb(all_tokens,
                                      torch.cat([torch.zeros([B, 1], dtype=bool, device=all_tokens.device),
                                                 keyframe_idx + 1],
                                                dim=1))
            all_tokens = self.encoder(src=all_tokens)
            output = all_tokens[:, 1:]
        elif self.arch == 2:
            output = self.decoder(tgt=self.keyframe_token.repeat([B, keyframe_idx.shape[1], 1]),
                                  memory=text_feat, memory_key_padding_mask=~text_mask)
        elif self.arch == 3:
            hidden_states = self.encoder(src=text_feat, src_key_padding_mask=~text_mask)
            output = self.decoder(tgt=self.keyframe_token.repeat([B, keyframe_idx.shape[1], 1]),
                                  memory=hidden_states, memory_key_padding_mask=~text_mask)

        if self.pose_only:
            pose = self.pose_head(output)
            time = None
        else:
            if self.head_type == 1:
                pose = self.pose_head(output)
                time = self.tpos_head(output)
            else:
                pose = self.pose_head(output[..., :output.shape[-1] // 2])
                time = self.tpos_head(output[..., output.shape[-1] // 2:])
        return pose, time

    def forward(self, text, keyframe_idx):
        if self.text_type == 2:
            return self.forward2(text, keyframe_idx)

        text_feat = self.text_proj(text['hidden'])
        text_mask = text['mask']
        B, L = text_feat.shape[0], text_feat.shape[1]

        if self.arch == 1:
            all_tokens = torch.cat([text_feat, self.keyframe_token.repeat([B, keyframe_idx.shape[1], 1])], dim=1)
            all_tokens = self.m_pos_emb(all_tokens,
                torch.cat([torch.zeros([B, L], dtype=bool, device=all_tokens.device), keyframe_idx+1], dim=1))

            all_masks = torch.cat([torch.ones([B, self.keyframe_token.shape[0]], dtype=bool,
                                              device=text_feat.device), text_mask], dim=1)
            all_tokens = self.encoder(src=all_tokens, src_key_padding_mask=~all_masks)
            output = all_tokens[:, L:]
        elif self.arch == 2:
            all_tokens = self.m_pos_emb(self.keyframe_token.repeat([B, keyframe_idx.shape[1], 1]), keyframe_idx)
            output = self.decoder(tgt=all_tokens, memory=text_feat, memory_key_padding_mask=~text_mask)
        elif self.arch == 3:
            hidden_states = self.encoder(src=text_feat, src_key_padding_mask=~text_mask)
            output = self.decoder(tgt=self.keyframe_token.repeat([B, 1, 1]),
                                     memory=hidden_states, memory_key_padding_mask=~text_mask)

        if self.pose_only:
            pose = self.pose_head(output)
            time = None
        else:
            if self.head_type == 1:
                pose = self.pose_head(output)
                time = self.tpos_head(output)
            else:
                pose = self.pose_head(output[..., :output.shape[-1] // 2])
                time = self.tpos_head(output[..., output.shape[-1] // 2:])

        return pose, time


class KFG1_df(nn.Module):
    def __init__(
        self,
        motion_dim=263,
        model_dim=512,
        text_feat_dim=512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 8,
        arch: int = 1,
        head_type=1,
        head_drop=0,
        text_type=1,
        pose_only=False,
    ):
        super().__init__()
        self.arch = arch
        self.head_type = head_type
        self.text_type = text_type
        self.pose_only = pose_only

        self.time_embed_dim = model_dim
        self.time_emb = TimestepEmbedding(model_dim, model_dim)
        self.pos_emb = PartialPositionEmbedding(512, model_dim, 0, grad=False)
        # self.pose_emb = PositionEmbedding(256, model_dim, 0)

        self.input_proj = nn.Linear(motion_dim, model_dim)

        if type(num_layers) is str:
            enc_layer, dec_layer = [int(x) for x in num_layers.split('-')]
        else:
            enc_layer, dec_layer = num_layers, num_layers

        self.text_proj = nn.Linear(text_feat_dim, model_dim)


        if head_type == 1 or self.pose_only:
            head_input = model_dim
        else:
            head_input = model_dim // 2
        self.pose_head = nn.Sequential(
            nn.Dropout(head_drop),
            nn.Linear(head_input, motion_dim, bias=True)
        )
        if not self.pose_only:
            self.tpos_head = nn.Sequential(
                nn.Dropout(head_drop),
                nn.Linear(head_input, 1, bias=False)
            )

        if self.arch == 1: # encoder only
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    model_dim, num_heads, dim_feedforward,
                    dropout, activation=F.gelu, batch_first=True,
                ), num_layers=enc_layer
            )
        elif self.arch == 2:    # decoder only
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward,
                                           dropout, activation=F.gelu, batch_first=True),
                num_layers=dec_layer
            )
        elif self.arch == 3:    # encoder-decoder
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    model_dim, num_heads, dim_feedforward,
                    dropout, activation=F.gelu, batch_first=True,
                ), num_layers=enc_layer
            )
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward,
                                           dropout, activation=F.gelu, batch_first=True),
                num_layers=dec_layer
            )

    def forward2(self, text, keyframe_noise, keyframe_idx, timestep):
        keyframe_noise = self.input_proj(keyframe_noise)
        text_feat = self.text_proj(text['text_emb'])
        time_embed = self.time_emb(timestep_embedding(timestep, self.time_embed_dim)).unsqueeze(dim=1)
        B = text_feat.shape[0]

        if self.arch == 1:
            all_tokens = torch.cat([time_embed, text_feat.unsqueeze(dim=1), keyframe_noise], dim=1)
            all_tokens = self.pos_emb(all_tokens,
                                        torch.cat([torch.zeros([B, 2], dtype=bool, device=all_tokens.device),
                                                   keyframe_idx + 2],
                                                  dim=1))
            # all_tokens = self.pose_emb(all_tokens)
            output = self.encoder(src=all_tokens)
            output = output[:, 2:]
        else:
            pass

        if self.pose_only:
            pose = self.pose_head(output)
            time = None
        else:
            if self.head_type == 1:
                pose = self.pose_head(output)
                time = self.tpos_head(output)
            else:
                pose = self.pose_head(output[..., :output.shape[-1] // 2])
                time = self.tpos_head(output[..., output.shape[-1] // 2:])
        return pose, time

    def forward(self, text, keyframe_noise, keyframe_idx, timestep):
        if self.text_type == 2:
            return self.forward2(text, keyframe_noise, keyframe_idx, timestep)

        keyframe_noise = self.input_proj(keyframe_noise)
        time_embed = self.time_emb(timestep_embedding(timestep, self.time_embed_dim)).unsqueeze(dim=1)
        text_feat = self.text_proj(text['hidden'])
        text_mask = text['mask']
        B = text_feat.shape[0]

        if self.arch == 1:
            all_tokens = torch.cat([time_embed, keyframe_noise, text_feat], dim=1)
            all_tokens = self.pose_emb(all_tokens)
            all_masks = torch.cat([torch.ones([B, 1+keyframe_noise.shape[1]], dtype=torch.bool,
                                            device=keyframe_noise.device), text_mask], dim=1)
            output = self.encoder(src=all_tokens, src_key_padding_mask=~all_masks)
            output = output[:, 1:keyframe_idx.shape[1]+1]

        elif self.arch == 2:
            all_tokens = torch.cat([time_embed, keyframe_noise], dim=1)
            output = self.decoder(tgt=all_tokens,
                         memory=text_feat, memory_key_padding_mask=~text_mask)
            output = output[:, 1:keyframe_idx.shape[1] + 1]
        elif self.arch == 3:
            all_tokens = torch.cat([time_embed, keyframe_noise], dim=1)
            hidden_states = self.encoder(src=text_feat, src_key_padding_mask=~text_mask)
            output = self.decoder(tgt=all_tokens,
                                     memory=hidden_states, memory_key_padding_mask=~text_mask)
            output = output[:, 1:keyframe_idx.shape[1]+1]


        if self.pose_only:
            pose = self.pose_head(output)
            time = None
        else:
            if self.head_type == 1:
                pose = self.pose_head(output)
                time = self.tpos_head(output)
            else:
                pose = self.pose_head(output[..., :output.shape[-1] // 2])
                time = self.tpos_head(output[..., output.shape[-1] // 2:])

        return pose, time
