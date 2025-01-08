import argparse
import os
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=888,type=int)
    parser.add_argument('--dataset', default='htm-pulseInter6-trans', type=str)
    parser.add_argument('--seq_len', default=2*60*5, type=int) #64
    parser.add_argument('--batch_size', default=24, type=int) 
    parser.add_argument('--MIL_k', default=2, type=int) #### 2
    parser.add_argument('--corr_thres', default=0.85, type=float) 
    parser.add_argument('--context_length', default=36, type=float) 
    parser.add_argument('--anatomy_aware', default=False, type=bool) 
    parser.add_argument('--with_alignability', default=False, type=bool) 
    parser.add_argument('--loss_correction', default=True, type=bool) #### True
    parser.add_argument('--mlm', default=True, type=bool) 
    parser.add_argument('--loss', default='nce', type=str)
    # parser.add_argument('--gt_type', default='sort', type=str) # sort, viterbi, split
    parser.add_argument('--lr', default=2e-6, type=float) #1e-4
    parser.add_argument('--wd', default=2e-7, type=float)
    parser.add_argument('--clip_grad', default=0.0, type=float) # 0.0 or 3.0
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('-j', '--num_workers', default=2, type=int)

    parser.add_argument('--test', default='', type=str)
    # parser.add_argument('--resume', default='/media/engs2527//2TBSlow1/log-tmp_step1/2024_12_10_16_01_htm-pulseInter6-trans_len600_pos-learned_bs24_MILk2_lr2e-06/model/epoch799.pth.tar', type=str)
    parser.add_argument('--resume', default='', type=str)
    # parser.add_argument('--pretrain', default='/media/engs2527/2TBFast/step1_1.pth.tar', type=str)
    # parser.add_argument('--pretrain', default='/media/engs2527/2TBSlow1/log-tmp_step1/2023_12_17_23_01_htm-pulseInter6-trans_len600_pos-learned_bs25_MILk2_lr2e-06/model/epoch611.pth.tar', type=str)
    parser.add_argument('--pretrain', default='', type=str)
    parser.add_argument('--epochs', default=2500, type=int) 
    parser.add_argument('--start_epoch', default=0, type=int)

    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('--prefix', default='tmp_step1', type=str)
    parser.add_argument('--backprop_freq', default=1, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--runtime_save_iter', default=200, type=int) 
    parser.add_argument('--optim_policy', default='default', type=str)

    parser.add_argument('--sim', default='cos', type=str)
    parser.add_argument('--aux_loss', default=1, type=int)
    parser.add_argument('--pos_enc', default='learned', type=str)
    parser.add_argument('--use_text_pos_enc', default=0, type=int)
    parser.add_argument('--loss_threshold', default=0.0, type=float)
    parser.add_argument('--learn_agreement', default=0, type=int)
    parser.add_argument('--temporal_agreement_type', default='keep', type=str)
    parser.add_argument('--use_alignability_head', default=0, type=int)
    parser.add_argument('--momentum_m', default=0.999, type=float)

    # transformer
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    

    args = parser.parse_args()
    return args


def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string

    if args.resume: 
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: 
        if os.path.dirname(args.test).endswith('model'):
            exp_path = os.path.dirname(os.path.dirname(args.test))
        else:
            exp_path = os.path.dirname(args.test)
    else:
        name_prefix = f"{args.name_prefix}_" if args.name_prefix else ""
        tag_agreement = f"_agree-{args.learn_agreement}-{args.temporal_agreement_type}" if args.learn_agreement else ""
        # exp_path = (f"log-{args.prefix}/{name_prefix}{dt_string}_"
        #     f"{args.model}_{args.loss}-th{args.loss_threshold}_{args.language_model}_{args.dataset}_len{args.seq_len}_"
        #     f"e{args.num_encoder_layers}d{args.num_decoder_layers}_pos-{args.pos_enc}_textpos-{args.use_text_pos_enc}_policy-{args.optim_policy}_"
        #     f"bs{args.batch_size}_lr{args.lr}{tag_agreement}")
        exp_path = (f"log-{args.prefix}/{name_prefix}{dt_string}_"
            f"{args.dataset}_len{args.seq_len}_"
            f"pos-{args.pos_enc}_"
            f"bs{args.batch_size}_MILk{args.MIL_k}_lr{args.lr}{tag_agreement}")

    exp_path = os.path.join('/media/engs2527//2TBSlow1/', exp_path)
    log_path = os.path.join(exp_path, 'log')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)

    with open(f'{log_path}/running_command.txt', 'a') as f:
        json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
        f.write('\n')

    return log_path, model_path, exp_path