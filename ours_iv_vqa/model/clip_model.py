import torch
from torch import nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence
from torch.nn import LayerNorm
from collections import OrderedDict
from transformers import BertModel, DistilBertModel
import numpy as np
from tfm_model import TemporalEncoder, get_position_embedding_sine
from word2vec_model import Word2VecModel
import open_clip
from collections import OrderedDict
from typing import Tuple, Union, Callable, Optional
import clip

class CLIPTextCfg:
    # bert_model_name: str = 'base'
    # context_length: int = 77
    vocab_size: int = 49409 #32000
    width: int = 512 #768
    heads: int = 8
    layers: int = 12
    fusion_layers: int = 1  # layers of fusion_module
    MOMENTUM: float = 0.5  # 0.99

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU,
            drop_attention_rate: float = 0.,
        ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=drop_attention_rate,
        )
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(
            self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU,
            drop_attention_rate: float = 0.,
        ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, drop_attention_rate=drop_attention_rate)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class TemporalAligner(nn.Module):
    def __init__(self, 
                 sim='cos', 
                 pos_enc='learned',
                 return_dual_feature=False,
                 random_pos_start=1,
                 ):
        super().__init__()
        self.sim = sim 
        self.pos_enc = pos_enc
        self.random_pos_start = random_pos_start

        ###################### text encoder ######################
        self.lang_model, preprocess = clip.load("ViT-B/32", device=device)
        self.tokenizer = clip.tokenize
        
        self.text_temporal_pos_embed = nn.Parameter(torch.empty(512, 512))
        nn.init.normal_(self.text_temporal_pos_embed, std=0.01)

        text_cfg = CLIPTextCfg
        self.fusion_module = Transformer(
            width=text_cfg.width,
            layers=text_cfg.fusion_layers,
            heads=text_cfg.heads,
        )
        self.img_special_token = nn.Parameter(torch.zeros(1, 1, 512))
        self.mlm_projection = nn.Parameter(torch.empty(text_cfg.width, text_cfg.vocab_size))
        nn.init.normal_(self.mlm_projection, std=text_cfg.width ** -0.5)


    def forward(self, video_embed, lang_embed, VQA=False,
                video_padding_mask=None, lang_padding_mask=None,
                # text_timestamp,
                interpolate_from=None,
                abs_text_pos=None,
                return_dual_feature=True):
        if not VQA:
            ### Dual Encoder ###
            B,T,_ = video_embed.shape
            contrastive_logits_dual = torch.einsum("astc,bkc->astbk", 
                video_embed.unsqueeze(dim=1), lang_embed)
            output_dict = {'logits_dual': contrastive_logits_dual}
            
            if return_dual_feature:
                feature_video_weight = torch.bmm(video_embed, lang_embed.permute(0,2,1))
                output_dict['similarity'] = feature_video_weight.permute(0,2,1) #torch.from_numpy(feature_video_weight.permute(0,2,1).detach().clone().cpu().numpy()).cuda() 
                feature_video_weight = torch.max(feature_video_weight, dim=-1)[0]
                feature_video_weight = torch.from_numpy(feature_video_weight.detach().clone().cpu().numpy()).cuda() 
                output_dict['feature_video_weight'] = feature_video_weight
                # output_dict['dual_feature_video'] = video_embed.unsqueeze(dim=1)
                # output_dict['dual_feature_text'] = lang_embed
            return output_dict
        else:
            B, _ = video_embed.shape
            # video_embed = self.img_proj(video_embed)
            img_special_tokens = self.img_special_token.expand(B, -1, -1)  # [128, 1, embed_dim]
            x = torch.cat([lang_embed, img_special_tokens, video_embed.unsqueeze(1)], dim=1) 
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.fusion_module(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x[:, :-3, :]  # Remove token [img_special_token, img]
            out = nn.LogSoftmax(dim=-1)(x @ self.mlm_projection)  # [batch_size=128, n_ctx=77, vocab_size=49409]
            return out


    def get_visual_feature(self, video_embed, video_padding_mask, interpolate_from=None):
        """Get video embedding from video transformer encoder in the dual model.
        No text inputs. Can be used for retrieval setting"""
        B,T,_ = video_embed.shape
        if interpolate_from:
            video_pos_embed_source = self.temporal_pos_embed[None, 0:interpolate_from, :]
            video_pos_embed = F.interpolate(video_pos_embed_source.transpose(1,2), 
                size=T, mode='linear', align_corners=False).transpose(1,2)
        else:
            if self.random_pos_start:
                pos_start_idx = np.random.randint(0, int(T/2))
            else:
                pos_start_idx = 0
            video_pos_embed = self.temporal_pos_embed[None, pos_start_idx:pos_start_idx+T, :]
        video_embed = video_embed + self.ln_position_init(video_pos_embed)

        return video_embed
    
    def get_text_visual_sim_dual(self, video_embed, lang_embed, interpolate_from=None):
        B,T,_ = video_embed.shape
        N = lang_embed.shape[1]
        video_padding_mask = torch.zeros(B,T,device=video_embed.device).bool()

        # video_out = self.get_visual_feature(
        #     video_embed, 
        #     video_padding_mask, 
        #     interpolate_from)
        # video_feature_norm = video_out / video_out.norm(dim=-1, keepdim=True)
        # video_feature_norm = video_feature_norm.unsqueeze(dim=1)

        contrastive_logits_dual = torch.einsum("bstc,bkc->bstk", 
            video_embed.unsqueeze(dim=1), lang_embed)

        return contrastive_logits_dual

class BinaryHeadWithPos(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(2, dim)
        self.linear = nn.Linear(2*dim, 1)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.normal_(self.linear.weight, std=0.01)
    
    def forward(self, x):
        x_value = x[...,0:self.dim]
        x_pos = self.proj(x[...,self.dim::])
        out = self.linear(torch.cat((x_value, x_pos), dim=-1))
        return out
