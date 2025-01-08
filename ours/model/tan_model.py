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
        model_name = 'PubMedBERT_256-timm-vit_base_patch16_224'
        checkpoint = '../../biomedclip/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/a0dc596371b78fb8c1d1d9dadaaad986e72dc731/models/2022_11_08-07_39_28-model_timm-vit_base_patch16_224-lr_0.0005-b_1024-j_8-p_amp/checkpoints/epoch_32.pt'

        self.lang_model = open_clip.create_model(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        checkpoint = torch.load(checkpoint, map_location="cpu")
        new_state_dict = OrderedDict()
        for k,v in checkpoint['state_dict'].items():
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
        self.lang_model.load_state_dict(new_state_dict, strict=False) # can set this to be true except for timm models

        # self.ln_position_init = LayerNorm(512)
        
        # temporal positional encoding for video
        # if self.pos_enc == 'learned':
        #     self.temporal_pos_embed = nn.Parameter(torch.empty(1536, 512))
        #     nn.init.normal_(self.temporal_pos_embed, std=0.01)
        # elif self.pos_enc == 'sine':
        #     temporal_pos_embed = get_position_embedding_sine(1536, 512)
        #     self.register_buffer('temporal_pos_embed', temporal_pos_embed)

        # temporal positional encoding for text
        # self.text_temporal_pos_embed = nn.Parameter(torch.empty(512, 512))
        # nn.init.normal_(self.text_temporal_pos_embed, std=0.01)

        ### classifier
        # self.img_special_token = nn.Parameter(torch.zeros(1, 1, 512))
        # self.encoder_mlm_projection = nn.Parameter(torch.empty(768, 49409))
        # nn.init.normal_(self.encoder_mlm_projection, std=768 ** -0.5)


    def forward(self, video_embed, lang_embed, 
                video_padding_mask, lang_padding_mask,
                # text_timestamp,
                interpolate_from=None,
                abs_text_pos=None,
                return_dual_feature=True,
                text_logits=False):
        if text_logits:
            output_dict = {}
            output_dict['text_logits'] = nn.LogSoftmax(dim=-1)(lang_embed @ self.encoder_mlm_projection)  # [batch_size=128, n_ctx=77, vocab_size=49409]
            return output_dict
        else:
            ### Dual Encoder ###
            # video attend itself with pos-enc
            B,T,_ = video_embed.shape

            # video_out = self.get_visual_feature(
            #     video_embed, 
            #     video_padding_mask, 
            #     interpolate_from)
            # video_feature_norm = video_out / video_out.norm(dim=-1, keepdim=True)
            # video_feature_norm = video_feature_norm.unsqueeze(dim=1)

            contrastive_logits_dual = torch.einsum("astc,bkc->astbk", 
                video_embed.unsqueeze(dim=1), lang_embed)

            output_dict = {'logits_dual': contrastive_logits_dual}
            
            if return_dual_feature:
                feature_video_weight = torch.bmm(video_embed, lang_embed.permute(0,2,1))
                feature_video_weight = torch.sigmoid(feature_video_weight)
                output_dict['similarity'] = feature_video_weight.permute(0,2,1) #torch.from_numpy(feature_video_weight.permute(0,2,1).detach().clone().cpu().numpy()).cuda() 
                feature_video_weight = torch.max(feature_video_weight, dim=-1)[0]
                feature_video_weight = torch.from_numpy(feature_video_weight.detach().clone().cpu().numpy()).cuda() 
                output_dict['feature_video_weight'] = feature_video_weight
                # output_dict['dual_feature_video'] = video_embed.unsqueeze(dim=1)
                # output_dict['dual_feature_text'] = lang_embed
            return output_dict

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
