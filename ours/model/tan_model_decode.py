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
# from modeling_decoding import BertForSeq2SeqDecoder, BertConfig

class CLIPTextCfg:
    # bert_model_name: str = 'base'
    # context_length: int = 77
    vocab_size: int = 30522 # 49409 #32000
    width: int = 768 #512 # 768
    heads: int = 12
    layers: int = 4
    in_token: int = 768
    types: int = 3
    fusion_layers: int = 4  # layers of fusion_module
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
            self, width: int, layers: int, heads: int, types: int, in_token: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU,
            drop_attention_rate: float = 0.,
        ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.type_embed = nn.Parameter(torch.empty(types, width))
        nn.init.normal_(self.type_embed, std=0.01)
        self.pos_embed = nn.Parameter(torch.empty(in_token, width))
        nn.init.normal_(self.pos_embed, std=0.01)
        # pos_embed = get_position_embedding_sine(1536, 512)
        # self.register_buffer('pos_embed', pos_embed)
        self.ln_init = LayerNorm(width)
        self.ln_type_init = LayerNorm(width)
        self.ln_position_init = LayerNorm(width)

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, drop_attention_rate=drop_attention_rate)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, lang_len, video_len, attn_mask: Optional[torch.Tensor] = None):
        type_embed1 = self.type_embed[0, None, :].repeat([lang_len, 1, 1])
        type_embed2 = self.type_embed[1, None, :].repeat([video_len+1, 1, 1])
        type_embed3 = self.type_embed[2, None, :].repeat([8, 1, 1])
        type_embed = torch.cat([type_embed1, type_embed2, type_embed3], dim=0)[:x.shape[0], :, :]
        pos_embed = self.pos_embed[:x.shape[0], None, :] # 62*bs*512
        # import pdb;pdb.set_trace()
        x = self.ln_init(x) + self.ln_position_init(pos_embed) + self.ln_type_init(type_embed) 
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask) # 62*bs*512
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

        ### image projection
        self.img_proj = nn.Linear(512, 768)
        nn.init.normal_(self.img_proj.weight, std=0.01)
        self.img_proj_clip = nn.Linear(512, 512)
        nn.init.normal_(self.img_proj_clip.weight, std=0.01)
        # self.img_proj_clip = ResidualAttentionBlock(512, 8, mlp_ratio=4.0, act_layer=nn.GELU, drop_attention_rate=0.0)

        ### multi-modal decoder
        text_cfg = CLIPTextCfg
        self.fusion_module = Transformer(
            width=text_cfg.width,
            layers=text_cfg.fusion_layers,
            heads=text_cfg.heads,
            types=text_cfg.types,
            in_token=text_cfg.in_token,
        )

        # self.fusion_module = open_clip.create_model(model_name)

        # self.fusion_module = BertForSeq2SeqDecoder.from_pretrained(
        #     '/home/engs2527/Downloads/unilm2-base-uncased.bin', config='/home/engs2527/Downloads/unilm2-base-uncased_config.json', mask_word_id=mask_word_id, search_beam_size=args.beam_size,
        #     length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
        #     forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
        #     ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
        #     max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift, 
        # )

        ### classifier
        self.img_special_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.mlm_projection = nn.Parameter(torch.empty(text_cfg.width, text_cfg.vocab_size))
        nn.init.normal_(self.mlm_projection, std=text_cfg.width ** -0.5)


    def forward(self, video_embed, lang_embed, knowledge=None, VQA=False,
                video_padding_mask=None, lang_padding_mask=None,
                # text_timestamp,
                interpolate_from=None,
                abs_text_pos=None,
                return_dual_feature=True):
        if not VQA:
            ### Dual Encoder ###
            B,T,D = video_embed.shape
            video_embed = video_embed + self.img_proj_clip(video_embed.reshape(B*T, D)).reshape(B, T, -1)
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
        else:
            B, T, D = video_embed.shape
            video_embed = self.img_proj(video_embed.reshape(B*T, D)).reshape(B, T, -1)
            img_special_tokens = self.img_special_token.expand(B, -1, -1)  # [128, 1, embed_dim]
            # import pdb;pdb.set_trace()
            if knowledge is None:
                x = torch.cat([lang_embed, img_special_tokens, video_embed], dim=1) 
            else:
                x = torch.cat([lang_embed, img_special_tokens, video_embed, knowledge], dim=1) 
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.fusion_module(x, lang_embed.shape[1], video_embed.shape[1])
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x[:, :lang_embed.shape[1], :]  # Remove token [img_special_token, img] bs * 60 *768
            out = nn.LogSoftmax(dim=-1)(x @ self.mlm_projection)  # [batch_size=128, n_ctx=77, vocab_size=49409]
            return out

    # def sequential_generation(self, model, input, max_len=15, leed_out_len=15, 
    #                         top_k=0, temperature=None, sample=True, cuda=False):
    #     """ Generate one word at a time, in L->R order """
    #     seed_len = input.shape[1]
    #     output = torch.zeros(input.shape)
        
    #     for ii in range(max_len):
    #         inp = [sent[:seed_len+ii+leed_out_len]+[sep_id] for sent in input]
    #         inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
    #         out = model(inp)
    #         idxs = generate_step(out, gen_idx=seed_len+ii, top_k=top_k, temperature=temperature, sample=sample)
    #         for jj in range(batch_size):
    #             batch[jj][seed_len+ii] = idxs[jj]
            
    #     return untokenize_batch(batch)


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
