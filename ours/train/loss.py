import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
import os
import sys
sys.path.append('..')
from utils.utils import get_youtube_link, second_to_time
import copy
from tqdm import tqdm
import ffmpeg
from torch.nn.utils.rnn import pad_sequence


def circulant(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.
    The additional axis is appended as the last dimension.
    E.g. 
    circulant(tensor([0,1,2]), dim=0) --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))


def get_mask_from_time(start_list, end_list, num_timestamp, num_text, k=0, device='cuda'):
    """get a binary mask of shape [Batchsize, Num_text, Time].
    For the n-th sentence in the b-th video, 
    the vector [1x1xTime] has value 1 if the text corresponds this time segment."""
    if k > 0:
        for i in range(len(end_list)):
            if len(end_list[i]) > k*2:
                end_list[i] = end_list[i][k:]
                end_list[i].extend([end_list[i][-1]]*k)

    B = len(start_list)
    steps = torch.arange(num_timestamp, device=device)[None,None,:].repeat(B, num_text, 1)
    start_list = pad_sequence(
        [torch.FloatTensor(i) for i in start_list],
        batch_first=True, 
        padding_value=num_timestamp+1e2).to(device, non_blocking=True)
    end_list = pad_sequence(
        [torch.FloatTensor(i) for i in end_list],
        batch_first=True, 
        padding_value=-1e2).to(device, non_blocking=True)
    mask = (start_list[:,:,None] <= steps) * (steps < end_list[:,:,None]) 
    return mask, start_list, end_list

def get_mask_from_time_anatomy(text_embed, start_list, end_list, num_timestamp, num_text, k = 0, tres=0.8, device='cuda'):
    """get a binary mask of shape [Batchsize, Num_text, Time].
    For the n-th sentence in the b-th video, 
    the vector [1x1xTime] has value 1 if the text corresponds this time segment."""
    if k > 0:
        for i in range(len(end_list)):
            if len(end_list[i]) > k*2:
                end_list[i] = end_list[i][k:]
                end_list[i].extend([end_list[i][-1]]*k)

    B = len(start_list)
    steps = torch.arange(num_timestamp, device=device)[None,None,:].repeat(B, num_text, 1)
    start_list = pad_sequence(
        [torch.FloatTensor(i) for i in start_list],
        batch_first=True, 
        padding_value=num_timestamp+1e2).to(device, non_blocking=True)
    end_list = pad_sequence(
        [torch.FloatTensor(i) for i in end_list],
        batch_first=True, 
        padding_value=-1e2).to(device, non_blocking=True)
    mask = (start_list[:,:,None] <= steps) * (steps < end_list[:,:,None]) 

    # save_matrix(mask.to(torch.float32), 'mask1')
    semantic_corr = torch.bmm(text_embed, text_embed.permute(0,2,1)) > tres
    # save_matrix(semantic_corr.to(torch.float32), 'semantic_corr')
    mask = torch.bmm(semantic_corr.to(torch.float32), mask.to(torch.float32))
    # save_matrix(mask.to(torch.float32), 'mask2')
    return mask, start_list, end_list


def get_text_pos(start_list, end_list, device='cuda'):
    B = len(start_list)
    start_list = pad_sequence(
        [torch.FloatTensor(i) for i in start_list],
        batch_first=True, padding_value=0).to(device, non_blocking=True)
    end_list = pad_sequence(
        [torch.FloatTensor(i) for i in end_list],
        batch_first=True, padding_value=0).to(device, non_blocking=True)
    return torch.stack((start_list, end_list), dim=-1)


def get_loss(input_data, video_seq, text_embed, video_padding_mask, text_padding_mask,
             logits, args, abs_text_pos, entity=False, GT=None, epoch=0):
    logits_dual = logits['logits_dual']
    alignability = input_data['align']

    if args.sim == 'cos':
        logits_dual = logits_dual / 0.07

    device = logits_dual.device
    B, T, _ = video_seq.shape
    N = text_embed.shape[1]
    num_enc_layers = logits_dual.shape[1]

    loss_dict = {}

    if entity:
        binary_tgt_raw = GT
    else:
        # binary tgt: B,T,B,N
        if not args.anatomy_aware:
            binary_tgt_raw, _, _ = get_mask_from_time(
                input_data['start'], input_data['end'],
                num_timestamp=T, num_text=N, k=args.MIL_k, device=device)  # B,N,T
        else:
            # introduce anatomy-aware contrastive learning
            binary_tgt_raw, _, _ = get_mask_from_time_anatomy(text_embed, 
                input_data['start'], input_data['end'],
                num_timestamp=T, num_text=N, k=args.MIL_k, tres=args.corr_thres, device=device)  # B,N,T
        loss_dict['binary_tgt_raw'] = binary_tgt_raw.detach()
        # introduce alignability in contrastive learning
        if args.with_alignability:
            binary_tgt_raw = binary_tgt_raw * alignability.unsqueeze(2).bool()
        # introduce loss correction
        if args.loss_correction:
            degree = epoch / float(args.epochs) 
            # import pdb;pdb.set_trace()
            mask = binary_tgt_raw.float().sum(2).unsqueeze(2)
            mask = torch.where(mask>0, torch.ones_like(mask), mask)
            similarity = logits['similarity'] * mask
            logits['similarity'] = similarity
            aa, bb, cc =  similarity.shape
            # print(mask.shape, mask[0], binary_tgt_raw.float()[0])[0]
            # loss_corr = 10 * (1 - degree) * torch.nn.MSELoss(reduce=True, reduction='mean')(similarity, binary_tgt_raw.float())
            loss_corr = torch.mean(10 * (1 - degree) * torch.nn.MSELoss(reduce=False)(similarity, binary_tgt_raw.float()) * (torch.ones_like(binary_tgt_raw.float()) + 2 * binary_tgt_raw.float()))
            loss_dict['loss-correction_simi'] = loss_corr.detach()
        
        if epoch > 200:
            rand = torch.from_numpy(np.random.binomial(1, degree, aa*bb*cc).reshape(aa, bb, cc)).to(device)
            binary_tgt_raw = (similarity > 0.5).bool() * rand + binary_tgt_raw * (1-rand).bool()
            binary_tgt_raw = binary_tgt_raw * mask

        # loss_dict['binary_tgt_raw'] = binary_tgt_raw.detach()
        
    binary_tgt = rearrange(binary_tgt_raw, 'b n t -> b t n').unsqueeze(2).repeat(1,1,B,1) * torch.eye(B, device=device)[:,None,:,None]
    loss_dual = Info_NCE_loss(binary_tgt, logits_dual, text_padding_mask, num_enc_layers, B, T)
    loss_dict['loss-dual'] = loss_dual.detach()

    '''
    if args.cross_img:
        # print(video_seq.shape, text_embed.shape, video_padding_mask.shape, text_padding_mask.shape)[0]
        # torch.Size([25, 600, 512]) torch.Size([25, 18, 512]) torch.Size([25, 600]) torch.Size([25, 18])
        idx = torch.randperm(B)
        video_seq_rand = video_seq[idx]
        idx = torch.randperm(B)
        text_embed_rand = text_embed[idx]
        video_lang_simi = torch.bmm(video_seq, text_embed.permute(0,2,1)) 
        videoR_lang_simi = torch.bmm(video_seq_rand, text_embed.permute(0,2,1))
        videoR_lang_simi = torch.from_numpy(videoR_lang_simi.detach().clone().cpu().numpy()).cuda() 
        video_langR_simi = torch.bmm(video_seq, text_embed_rand.permute(0,2,1))
        video_langR_simi = torch.from_numpy(video_langR_simi.detach().clone().cpu().numpy()).cuda() 

        video_videoR_simi = torch.bmm(video_seq, video_seq_rand.permute(0,2,1))
        video_videoR_simi1 = torch.bmm(video_videoR_simi.permute(0,2,1), video_lang_simi)
        if np.random.uniform() > 0.95:
            save_matrix(videoR_lang_simi.to(torch.float32), 'videoR_lang_simi')
            save_matrix(video_videoR_simi1.to(torch.float32), 'video_videoR_simi1')
        # print(video_videoR_simi.max(), video_videoR_simi.min())
        # print(video_videoR_simi1.max(), video_videoR_simi1.min())

        langR_langR_simi = torch.bmm(text_embed, text_embed_rand.permute(0,2,1))
        langR_langR_simi1 = torch.bmm(video_lang_simi, langR_langR_simi.permute(0,2,1))
        if np.random.uniform() > 0.95:
            save_matrix(video_langR_simi.to(torch.float32), 'video_langR_simi')
            save_matrix(langR_langR_simi1.to(torch.float32), 'langR_langR_simi1')
        # print(langR_langR_simi.max(), langR_langR_simi.min())
        # print(langR_langR_simi1.max(), langR_langR_simi1.min())

        loss_mse = torch.nn.MSELoss(reduce=True, reduction='mean')
        cross_img_loss = loss_mse(videoR_lang_simi, video_videoR_simi1) + loss_mse(video_langR_simi, langR_langR_simi1)
        loss_dict['loss-cross_img'] = cross_img_loss.detach()   
        # print(loss_mse(videoR_lang_simi, video_videoR_simi1))[0] 
        '''    

    ### compute the final loss ###
    if entity:
        loss_dict['loss'] = loss_dual
    else:
        if args.loss_correction:
            loss_dict['loss'] = loss_dual + loss_corr
        else:
            loss_dict['loss'] = loss_dual
    
    ### visualization (optional) ###
    if False: # args.pretrain: # temporary, for debug
        logits_dual_vis = logits_dual[:,-1,:]
        idx = 0
        visualize(logits_dual_vis * 0.07, binary_tgt, 
            input_data['text'], input_data['vid'],
            input_data['start'], input_data['end'], 
            'dual', idx, args)
        import ipdb; ipdb.set_trace()

    return loss_dict

def norm_simi(M, dim=-1):
    return M / (M.norm(dim=dim, keepdim=True) + 1e-10)

def Info_NCE_loss(binary_tgt, logits_dual, text_padding_mask, num_enc_layers, B, T):
    ### prepare tgt ###
    no_padding_binary_tgt = binary_tgt[:,:,~text_padding_mask.bool()]
    no_padding_binary_tgt = no_padding_binary_tgt.view(B*T,-1)
    video_mask_with_pos = no_padding_binary_tgt.sum(-1) > 0
    text_mask_with_pos = no_padding_binary_tgt.sum(-2) > 0

    ### get logits for dual model ###
    no_padding_logits_dual = logits_dual[:,:,:,~text_padding_mask.bool()]
    no_padding_logits_dual = no_padding_logits_dual.permute(1,0,2,3).reshape(num_enc_layers, B*T, -1)
    
    no_padding_logits_dual_pos = no_padding_logits_dual.clone()
    no_padding_logits_dual_pos[:,~no_padding_binary_tgt.bool()] = - 6e4
    no_padding_logits_dual_neg = no_padding_logits_dual

    v_numerator_dual = torch.logsumexp(no_padding_logits_dual_pos, dim=-1)
    v_denomenator_dual = torch.logsumexp(no_padding_logits_dual_neg, dim=-1)
    v_loss_milnce_dual = (v_denomenator_dual - v_numerator_dual)[:,video_mask_with_pos.bool()]
    
    t_numerator_dual = torch.logsumexp(no_padding_logits_dual_pos, dim=-2)
    t_denomenator_dual = torch.logsumexp(no_padding_logits_dual_neg, dim=-2)
    t_loss_milnce_dual = (t_denomenator_dual - t_numerator_dual)[:,text_mask_with_pos.bool()]

    loss_dual = (v_loss_milnce_dual.mean() + t_loss_milnce_dual.mean()) / 2
    return loss_dual

def get_coarse_loss(para_embed, whole_video_embed):
    device = whole_video_embed.device

    mm = para_embed.mm(whole_video_embed.permute(1,0)) #/ 0.2
    pos = torch.eye(whole_video_embed.shape[0], device=device)
    pos = torch.where(pos>0, mm, - 6e4)

    loss_v = (torch.logsumexp(mm, dim=1) - torch.logsumexp(pos, dim=1)).mean()
    loss_t = (torch.logsumexp(mm, dim=0) - torch.logsumexp(pos, dim=0)).mean()

    return (loss_v + loss_t) / 2.

def save_matrix(mat, name):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import gc
    dual_sim = mat.data.cpu().numpy()[0,:,:]
    # plt.rcParams['figure.figsize'] = dual_sim.shape[1], dual_sim.shape[0]
    fig, ax = plt.subplots(1,1)
    ax.matshow(dual_sim, interpolation=None, aspect='auto')
    plt.savefig('./vis/'+name+'_vis.png')
    plt.cla()
    plt.clf()
    plt.close('all')   
    plt.close(fig)
    gc.collect()

def visualize(raw_logits, binary_tgt, sentences, vids, starts, ends, name_tag, idx, args, 
              num_vis_sample=2, start_ts=0, alignability_gt=None, alignability_pred=None):
    # except cos similarity
    raw_logits = raw_logits.float().detach().cpu()
    binary_tgt = binary_tgt.detach().cpu()
    if 'shift' in name_tag:
        title = 'Shifted-GT'
    else:
        title = 'GT'

    figsize = (16,12)
    fig, axes = plt.subplots(num_vis_sample*2,1,figsize=figsize)  # 16,12
    with torch.no_grad():
        for b_idx in range(num_vis_sample):
            start_ = starts[b_idx]
            end_ = ends[b_idx]
            vid_ = vids[b_idx]
            sent_ = sentences[b_idx]
            num_sent = len(sent_)
            if raw_logits.dim() == 4:
                logits_ = raw_logits[b_idx, :, b_idx, :][:, 0:num_sent].transpose(0,1)
            else:
                logits_ = raw_logits[b_idx, :, 0:num_sent].transpose(0,1)
            if binary_tgt.dim() == 4:
                tgt_ = binary_tgt[b_idx, :, b_idx, :][:, 0:num_sent].transpose(0,1)
            elif binary_tgt.dim() == 3:
                tgt_ = binary_tgt[b_idx, :, :][:, 0:num_sent].transpose(0,1)
            else:
                raise NotImplementedError(f"dim:{binary_tgt.dims()} is not supported")
            ratio = 3
            height_ = num_sent * ratio
            logits_interpolate = F.interpolate(logits_[None,None,:,],
                size=(height_, logits_.shape[1]), mode='nearest')[0,0]
            tgt_interpolate = F.interpolate(tgt_[None,None,:,],
                size=(height_, logits_.shape[1]), mode='nearest')[0,0]
            
            tmp = []
            for s in sent_:
                if len(s) < 48:
                    tmp.append(s)
                else:
                    tmp.append(s[0:48]+'...')
            sent_ = tmp
            if alignability_gt is not None:
                sent_suffix_ = []
                for s, a in zip(sent_, alignability_gt):
                    if a:
                        sent_suffix_.append(s+"[{}]".format('\u2714'))
                    else:
                        sent_suffix_.append(s+"[{}]".format('\u2718'))
            else:
                sent_suffix_ = sent_

            if alignability_pred is not None:
                sent_suffix_pred_ = []
                for s, a in zip(sent_, alignability_pred):
                    if a:
                        sent_suffix_pred_.append(s+"[{}]".format('\u2714'))
                    else:
                        sent_suffix_pred_.append(s+"[{}]".format('\u2718'))
            else:
                sent_suffix_pred_ = sent_

            sent_ticks = np.arange(num_sent) * ratio + ratio/2 - 0.5

            time_ticks = np.arange(0,64+1,8) + start_ts
            time_ticks = second_to_time(time_ticks)

            axes[b_idx * 2].imshow(tgt_interpolate.numpy())
            axes[b_idx * 2].set_yticks(sent_ticks)
            axes[b_idx * 2].set_yticklabels(sent_suffix_)
            # axes[b_idx * 2].set_title(f'{title} for {vid_} from {start_}s to {end_}s')
            axes[b_idx * 2].set_xticks(np.arange(0,64+1,8)-0.5); axes[b_idx * 2].set_xticklabels(time_ticks)
            axes[b_idx * 2].grid(which='major', axis='x', linestyle='--')
            # axp = axes[b_idx * 2 + 1].imshow((logits_interpolate.numpy() + 1) / 2,)
            axp = axes[b_idx * 2 + 1].imshow(logits_interpolate.numpy(),)
            arg_max = logits_.argmax(-1)
            # axes[b_idx * 2 + 1].set_title(f'Pred for {vid_} from {start_}s to {end_}s\n'
            #     # f'Max at {arg_max}'
            #     )
            axes[b_idx * 2 + 1].set_yticks(sent_ticks)
            axes[b_idx * 2 + 1].set_yticklabels(sent_suffix_pred_)
            axes[b_idx * 2 + 1].set_xticks(np.arange(0,64+1,8)-0.5); axes[b_idx * 2 + 1].set_xticklabels(time_ticks)
            axes[b_idx * 2 + 1].grid(which='major', axis='x', linestyle='--')
            # cb = plt.colorbar(axp, ax=[axes[b_idx * 2 + 1]])
    
    plt.savefig(os.path.join(args.log_path, f'iter-{idx:02d}_{vid_}_{name_tag}.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    return 