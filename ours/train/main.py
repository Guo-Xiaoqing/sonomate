import os
import sys
import torch
from torch.utils import data 
from transformers import BertModel, BertTokenizer
from tensorboardX import SummaryWriter
import numpy as np 
import random 
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import time
import math
import functools
import torch.cuda.amp as amp 
from config import parse_args, set_path
from loss import get_loss, get_coarse_loss, get_mask_from_time, get_text_pos
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import gc
import torch.nn.functional as F

sys.path.append('../data/')
from data.loader_htm import Pulse_FeatureLoader, pad_sequence_by_last
sys.path.append('../model/')
from tan_model_decode import TemporalAligner
from word2vec_model import Word2VecTokenizer
sys.path.append('../')
import utils.tensorboard_utils as TB
from utils.data_utils import DataLoaderBG
from utils.train_utils import clip_gradients
from utils.utils import AverageMeter, save_checkpoint, neq_load_customized, \
calc_topk_accuracy, ProgressMeter, neq_load_customized, save_runtime_checkpoint
from eval.eval_zeroshot_align import test_alignment_pulse 
from eval.eval_zeroshot_1stclassification import test_1stclass_pulse
# from eval.eval_zeroshot_1stclassification_small import test_1stclass_pulse_small
from eval.eval_zeroshot_2ndclassification import test_2ndclass_pulse
# from eval.eval_zeroshot_2ndclassification_val import test_2ndclass_pulse_val
from eval.eval_zeroshot_external import test_external

def token_mask(token, mask_rate=0.5): 
    mask = (torch.rand(token.shape) > mask_rate).type(torch.FloatTensor).cuda()
    token_mask = token * mask + (torch.ones_like(mask) - mask) * 4
    token_mask = token_mask.type(torch.IntTensor).cuda()
    token_mask = torch.where(token==0, token, token_mask)
    token_mask = torch.where(token==2, token, token_mask)
    token_mask = torch.where(token==3, token, token_mask)
    return token_mask

def train(loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    progress = ProgressMeter(
        len(loader), [batch_time, data_time, losses],
        prefix='Epoch:[{}]'.format(epoch))
    model.train()

    end = time.time()
    tic = time.time()
    optimizer.zero_grad()

    for idx, input_data in enumerate(loader):
        data_time.update(time.time() - end)
        video_seq = input_data['video'].to(device, non_blocking=True)
        video_padding_mask = input_data['padding_mask'].to(device, non_blocking=True)

        ########## get sentence feature w/ CLIP image and text encoder
        # get text embedding
        tokens = []
        entities = []
        paragraph = []
        tokens_para = []
        for sen in input_data['text']:
            token = model.tokenizer(sen, context_length=args.context_length).to(device)
            tokens.append(token)
        for sen in input_data['entity']:
            token = model.tokenizer(sen, context_length=args.context_length).to(device)
            entities.append(token)
            #### obtain paragraph embedding
            parai = ''
            for seni in sen:
                parai = parai + seni + ' '
            paragraph.append(parai)
        tokens_para = model.tokenizer(paragraph, context_length=512).to(device)
        para_embed, _ = model.lang_model(tokens_para.long())
        # import pdb;pdb.set_trace()
        tokens_para = tokens_para[:, :128]
        tokens_para_mask = token_mask(tokens_para, mask_rate=0.5)
        para_embed_token, _ = model.lang_model(tokens_para_mask, if_token=True)
        ####################################################

        input_data['token'] = tokens

        num_sentence_per_sample = [i.shape[0] for i in input_data['token']]
        token_list = [i.to(device, non_blocking=True) for i in input_data['token']]
        flatten_sentence_token = torch.concat(token_list, 0)
        token_list = [i.to(device, non_blocking=True) for i in entities]
        flatten_entity_token = torch.concat(token_list, 0)
        
        # get per-sentence feature
        flatten_sentence_token = flatten_sentence_token.long()
        flatten_entity_token = flatten_entity_token.long()
        text_embed, _ = model.lang_model(flatten_sentence_token)        
        entity_embed, _ = model.lang_model(flatten_entity_token)
        text_embed = pad_sequence_by_last(torch.split(text_embed, num_sentence_per_sample, dim=0))
        entity_embed = pad_sequence_by_last(torch.split(entity_embed, num_sentence_per_sample, dim=0))
        text_padding_mask = pad_sequence(torch.split(
            torch.zeros(flatten_sentence_token.shape[0], device=device),
            num_sentence_per_sample, dim=0), 
            batch_first=True, padding_value=1)
        del token_list, flatten_sentence_token

        # get alignability score for each sentence
        # sen_num = text_embed.shape[1]
        # input_data['alignM'] = np.zeros((text_embed.shape[0], text_embed.shape[1]))
        # for m in range(text_embed.shape[0]):
        #     input_data['alignM'][m, :len(input_data['align'][m])] = np.array(input_data['align'][m])
        # input_data['align'] = torch.from_numpy(input_data['alignM']).to(device)

        # get text timestamp for training
        B, T, _ = video_seq.shape
        N = text_embed.shape[1]

        # binary_sentence_timestamp, start_tensor, end_tensor = get_mask_from_time(
        #     input_data['start'], input_data['end'], 
        #     num_timestamp=T, num_text=N, device=device)

        if 'abs_text_start' in input_data:
            abs_text_pos = get_text_pos(input_data['abs_text_start'], input_data['abs_text_end'])
        else:
            abs_text_pos = None

        # forward pass
        with amp.autocast():
            logits = model(video_seq, text_embed, 
                    video_padding_mask=video_padding_mask.bool(), 
                    lang_padding_mask=text_padding_mask.bool(),
                    # text_timestamp=binary_sentence_timestamp,
                    abs_text_pos=abs_text_pos,
                    )
            feature_video_weight = logits['feature_video_weight'].unsqueeze(dim=2)
            whole_video_embed = torch.sum(video_seq * feature_video_weight, dim=1) / (torch.sum(feature_video_weight, dim=1) + 10e-6)

            loss_dict = get_loss(input_data=input_data, 
                                 video_seq=video_seq, 
                                 text_embed=text_embed, 
                                 video_padding_mask=video_padding_mask, 
                                 text_padding_mask=text_padding_mask,
                                 logits=logits, 
                                 args=args,
                                 abs_text_pos=abs_text_pos,
                                 epoch = epoch)

            coarse_loss = get_coarse_loss(para_embed, whole_video_embed.to(device))
            loss_dict['coarse_loss'] = coarse_loss

            outputs = model(video_seq[:, :int(video_seq.shape[1]/4), :], para_embed_token, VQA=True)
            loss_mlm = F.nll_loss(outputs.transpose(1, 2), tokens_para, ignore_index=0)#, reduce=False)
            loss_dict['mlm-loss'] = loss_mlm                    
                    
            # ######## similarity matrix visualization
            # for visi in range(logits['similarity'].shape[0]):
            # # for i in range(1):
            #     if len(input_data['text'][visi]) < 10 and len(input_data['text'][visi]) > 2:
            #         # tgt_newvis = (logits['similarity'] > 0.5).bool().data.cpu().numpy()[0,:,:]
            #         tgt_newvis = logits['similarity'].data.cpu().numpy()[visi,:,:]
            #         tgtvis = loss_dict['binary_tgt_raw'].bool().data.cpu().numpy()[visi,:,:]
            #         bbb = np.ones(tgtvis.shape)[:len(input_data['text'][visi])]
            #         # import pdb;pdb.set_trace()
            #         # print(np.unique(tgt_newvis), np.unique(tgtvis))
            #         simi_save(tgt_newvis, input_data['text'][visi], prefix = 'new')
            #         simi_save(tgtvis, input_data['text'][visi], prefix = '')
            #         simi_save(bbb, input_data['text'][visi], if_text=True, prefix = 'text')
            #         # del loss_dict['binary_tgt_raw']
            # ############################################

            logits = model(video_seq, entity_embed, 
                    video_padding_mask=video_padding_mask.bool(), 
                    lang_padding_mask=text_padding_mask.bool(),
                    # text_timestamp=binary_sentence_timestamp,
                    abs_text_pos=abs_text_pos,
                    return_dual_feature = True
                    )
            loss_dict1 = get_loss(input_data=input_data, 
                                 video_seq=video_seq, 
                                 text_embed=entity_embed, 
                                 video_padding_mask=video_padding_mask, 
                                 text_padding_mask=text_padding_mask,
                                 logits=logits, 
                                 args=args,
                                 abs_text_pos=abs_text_pos,
                                 entity = True,
                                 GT = loss_dict['binary_tgt_raw'],
                                 epoch = epoch)
            loss_dict['entity_loss'] = loss_dict1['loss']
            del loss_dict['binary_tgt_raw']

        loss = loss_dict['coarse_loss'] + loss_dict['loss'] + loss_dict1['loss'] + 0.1 * loss_dict['mlm-loss'] 
        if (not torch.isinf(loss)) and (not torch.isnan(loss)):
            losses.update(loss.item(), B)

        # backward pass
        grad_scaler.scale(loss).backward()
        if idx % args.backprop_freq == 0:
            grad_scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                _ = clip_gradients(model, clip_grad=args.clip_grad)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        # log stats
        if args.iteration % 5 == 0:
            for k, v in loss_dict.items():
                args.train_plotter.add_data(f'local/{k}', v.item(), args.iteration)
            args.train_plotter.add_data('local/lr', lr_scheduler.get_last_lr()[0], args.iteration)
            args.train_plotter.add_data('device/sps', 1/(time.time()-end), args.iteration)
            args.train_plotter.log_gpustat(step=args.iteration)
            args.train_plotter.writer.flush()

        if args.prof is not None:
            args.prof.step()

        batch_time.update(time.time() - end)
        progress.display(idx)
        lr_scheduler.step(args.iteration)
        end = time.time()
        args.iteration += 1

        # save runtime ckpt (for long-schedule training)
        if args.iteration % args.runtime_save_iter == 0:
            print('saving runtime checkpoint ...')
            state_dict = model.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': 1e5,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_runtime_checkpoint(save_dict, 
                filename=os.path.join(args.model_path, 'runtime.pth.tar'))

            _ = evaluate(loader, model, device, epoch, args)
            model.train()

    print(f'epoch {epoch} finished, takes {time.time() - tic} seconds')
    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    return losses.avg

def simi_save(input, tinput, if_text= False, prefix = None):
    # print(np.unique(np.uint8(input) * 255))
    plt.rcParams['figure.figsize'] = 10, len(tinput)
    fig, ax = plt.subplots(1,1)
    mat = np.uint8(input * 255.)[:len(tinput)]
    # mat = np.uint8(input * 255.)
    # mat = mat[np.nonzero(np.sum(mat, axis=1))]
    if if_text:
        # if prefix == 'text':
        for t in range(len(tinput)):
            ax.text(0, t+0.1, tinput[t], ha='left', wrap=True, size=12)
        plt.axis('off')
        # else:
        #     mat = 255 - mat
        ax.matshow(mat, aspect='auto', cmap = 'binary')
    else:
        ax.matshow(mat, aspect='auto')#, cmap = 'summer')
    # import pdb;pdb.set_trace()
    # ax.set_ylabel('Dual', fontweight ='bold')
    # ax.tick_params(axis="x", labelsize=6)
    # ax.tick_params(axis="y", labelsize=6)
    plt.savefig('./vis/'+str(len(tinput))+tinput[0][:1]+'_vis'+prefix+'.png')
    plt.cla()
    plt.clf()
    plt.close('all')   
    plt.close(fig)
    gc.collect()

@torch.no_grad()
def evaluate_downstream(model, device, args, epoch=0):
    model.eval()  # remember to change back during training
    all_metrics = {}

    ### alignment task on Pulse-Align ###
    def get_text_visual_sim(video_embed, text_str, save=False, interpolate_from=None, abs_text_pos=None):
        text_token = model.tokenizer(text_str, context_length=args.context_length).to(device)
        text_embed, _ = model.lang_model(text_token)
        del text_token
       
        # test alignment with dual model (optional):
        dual_logits = model.get_text_visual_sim_dual(video_embed, text_embed[None,:], interpolate_from)
        
        out_dict = {'dual-sim': dual_logits.transpose(-1,-2) / 0.07
                    }  # expect B,S,K,T

        ##### Save align matrix #####
        save = True
        if save:
            dual_sim = out_dict['dual-sim'].data.cpu().numpy()[0,-1,:,:]
            plt.rcParams['figure.figsize'] = 10, len(text_str)/2+1
            fig, ax = plt.subplots(1,1)
            ax.matshow(dual_sim, interpolation=None, aspect='auto')
            for t in range(len(text_str)):
                ax.text(0, t+0.1, text_str[t], ha='left', wrap=True)
            ax.set_ylabel('Dual', fontweight ='bold')
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)
            plt.savefig('./vis/'+text_str[t][0:5]+'_vis.png')
            plt.cla()
            plt.clf()
            plt.close('all')   
            plt.close(fig)
            gc.collect()
            del dual_sim
        ##################################
        return out_dict  

    accuracy_1st = test_1stclass_pulse(model, device, args)
    all_metrics.update({ 
        'pulse-1stclass': accuracy_1st})
    accuracy_2nd = test_2ndclass_pulse(model, device, args)
    all_metrics.update({ 
        'pulse-2ndclass': accuracy_2nd})
    accuracy_external = test_external(model, device, args)
    all_metrics.update({ 
        'pulse-external': accuracy_external})
    # xiaoqing
    # accuracy_1st_small = test_1stclass_pulse_small(model, device, args)
    # all_metrics.update({ 
    #     'pulse-1stclass_small': accuracy_1st_small})
    # accuracy_2nd_val = test_2ndclass_pulse_val(model, device, args)
    # all_metrics.update({ 
    #     'pulse-2ndclass_val': accuracy_2nd_val})
    # pulse_align_metrics = test_alignment_pulse(get_text_visual_sim, device, args, epoch=epoch)
    # all_metrics.update({ 
    #         'pulseAlign-R1': pulse_align_metrics['Recall'],
    #         'pulseAlign-AUC': pulse_align_metrics['AUC']})

    model.train()
    if args.optim_policy == 'bce':  # skip YC2
        return all_metrics

    return all_metrics


@torch.no_grad()
def evaluate(loader, model, device, epoch, args):
    model.eval()
    metric_dict = evaluate_downstream(model, device, args, epoch=epoch)
    for k, v in metric_dict.items():
        args.val_plotter.add_data(f'metric/{k}', v.item(), epoch)
    
    return metric_dict['pulse-2ndclass'].item()


def setup(args):
    # DDP setting (not using DDP in our exp)
    args.distributed = int(os.environ.get('SLURM_JOB_NUM_NODES', "1")) > 1

    # CUDA setting
    if torch.cuda.is_available():
        if args.gpu is None:
            args.gpu = str(os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
        device = torch.device('cuda')

        num_gpu = len(str(args.gpu).split(','))
        args.num_gpu = num_gpu
        args.batch_size = num_gpu * args.batch_size
        print('=> Effective BatchSize = %d' % args.batch_size)
    else:
        args.num_gpu = 0
        device = torch.device('cpu')
        print('=> Run with CPU')

    # general setting
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.iteration = 1
    args.log_path, args.model_path, args.exp_path = set_path(args)

    # tensorboard monitor in the background threads
    writer_train = SummaryWriter(logdir=os.path.join(args.log_path, 'train'), flush_secs=60)
    args.train_plotter = TB.PlotterThread(writer_train)
    writer_val = SummaryWriter(logdir=os.path.join(args.log_path, 'val'), flush_secs=60)
    args.val_plotter = TB.PlotterThread(writer_val)

    args.tokenizer = Word2VecTokenizer()
    return device

def get_dataset(args):
    tokenizer = args.tokenizer
    D = Pulse_FeatureLoader
    train_dataset = D(
        text_tag=args.dataset,
        tokenizer=tokenizer,
        mode='train',
        duration=args.seq_len)
    val_dataset = D(
        text_tag=args.dataset,
        tokenizer=tokenizer,
        mode='val',
        duration=args.seq_len)

    train_sampler = data.RandomSampler(train_dataset)
    val_sampler = data.SequentialSampler(val_dataset) 

    train_loader = DataLoaderBG(train_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn, pin_memory=True, drop_last=True,
        shuffle=(train_sampler is None), sampler=train_sampler, 
    )

    val_loader = DataLoaderBG(val_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=False,
        shuffle=(val_sampler is None), sampler=val_sampler, 
    )

    return train_dataset, val_dataset, train_loader, val_loader


def optim_policy(model, args, policy='default'):
    params = []
    no_decay = ['.ln_', '.bias', '.logit_scale', '.entropy_scale']
    param_group_no_decay = []
    param_group_with_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if policy == 'default':
            if any([i in name for i in no_decay]):
                param_group_no_decay.append(param)
            else:
                param_group_with_decay.append(param)
        elif policy == 'bce':
            if 'binary_head' in name:
                if any([i in name for i in no_decay]):
                    param_group_no_decay.append(param)
                else:
                    param_group_with_decay.append(param)
            else:
                param.requires_grad = False
                continue

    params.append({'params': param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
    params.append({'params': param_group_with_decay, 'lr': args.lr, 'weight_decay': args.wd})
    return params


def main(args):
    # pre-setup: overwritting
    device = setup(args)
    train_dataset, val_dataset, train_loader, val_loader = get_dataset(args)

    ### Model ###
    model = TemporalAligner(
                        sim=args.sim,
                        pos_enc=args.pos_enc,
                        return_dual_feature=True,
        )

    model.to(device)
    model_without_dp = model

    # for i, para in enumerate(model.named_parameters()):
    #     (name, param) = para
    #     print(i, name)	
    # print(i, name)[0]	

    ### optimizer ###
    params = optim_policy(model, args, args.optim_policy)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if not args.test:
        print('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)
        print('=================================\n')

    ### test ###
    if args.test:
        print('### test on downstream tasks ###')
        if args.test.lower() == 'random':
            print("[Warning] testing random weights")
        else:
            args.test = get_model_card(args.test)
            checkpoint = torch.load(args.test, map_location='cpu')
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            try:
                model_without_dp.load_state_dict(state_dict)
            except:
                model_without_dp.load_state_dict(state_dict, strict=False)
                print('[WARNING] Non-Equal load for testing!')

        model.eval()
        args.downstream = 1

        if args.downstream:
            metric_dict = evaluate_downstream(model, device, args)
        else:
            val_loss = evaluate(val_loader, model, device, epoch, args)
        sys.exit(0)

    ### restart ###
    best_acc = 1e5 
    if args.resume:
        print(f"resume from checkpoint {args.resume}")
        args.resume = get_model_card(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args.start_epoch = checkpoint['epoch']+1
        args.iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
            user_input = input('[WARNING] Non-Equal load for resuming training! continue? [y/n]')
            if user_input.lower() == 'n': sys.exit()
        optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.pretrain:
        print(f"pretrain from checkpoint {args.pretrain}")
        args.pretrain = get_model_card(args.pretrain)
        checkpoint = torch.load(get_model_card(args.pretrain), map_location='cpu')
        state_dict = checkpoint['state_dict']

        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
            ######## user_input = input('[WARNING] Non-Equal load for resuming training! continue? [y/n]')
            ######## if user_input.lower() == 'n': sys.exit()
        
    args.decay_steps = args.epochs * len(train_loader)
    args.warmup_iterations = 1000
    def lr_schedule_fn(iteration, iter_per_epoch, args):
        if iteration < args.warmup_iterations:
            lr_multiplier = iteration / (args.warmup_iterations)
        else:
            lr_multiplier = 0.5 * \
                (1. + math.cos(math.pi * (iteration - args.warmup_iterations) / (args.epochs*iter_per_epoch - args.warmup_iterations)))
        # lr_multiplier = (args.decay_steps - iteration) / args.decay_steps
        return lr_multiplier

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(lr_schedule_fn, iter_per_epoch=len(train_loader), args=args)
    )
    lr_scheduler.step(args.iteration)  # for resume mode
    grad_scaler = amp.GradScaler()

    # profiler, optional
    args.prof = None
    
    print('Main loop starts')
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        train_loss = train(train_loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args)
        # _ = evaluate(val_loader, model, device, epoch, args)

        if (epoch % args.eval_freq == 0) or (epoch == args.epochs - 1):
            is_best = train_loss < best_acc  # temporary use val loss
            best_acc = min(train_loss, best_acc)
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_checkpoint(save_dict, is_best, args.eval_freq, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=False) 

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    sys.exit(0)


def get_model_card(tag):
    """allow saving ckpt shortcuts in model_card_dict. """
    model_card_dict = {}
    if tag in model_card_dict:
        print(f'getting model tag {tag}: {model_card_dict[tag]}')
    return model_card_dict.get(tag, tag)


if __name__ == '__main__':
    args = parse_args()
    main(args)

"""
python main.py --model init --dataset htm-370k --batch_size 128 --use_text_pos_enc 0 --epochs 20
python main.py --model cotrain --dataset htm-370k --batch_size 128 --use_text_pos_enc 0 --epochs 20 --pretrain {} --loss_threshold 0.5
"""