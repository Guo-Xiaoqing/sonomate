import os
import numpy as np
import glob
from collections import OrderedDict

import torch
from PIL import Image
import open_clip
from torch.utils import data
from tqdm import tqdm 
from sklearn import metrics 
from utils.data_utils import DataLoaderFast, DataLoaderBG
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from data.loader_medvideoqa_pad import Binary_VQA_Dataset, Binary_VQA_Dataset_test
import difflib 
import random 
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

anatomy = {
        'brain': ['head circumference', 'brain', 'biparietal diameter', 'suboccipitobregmatic', 'cerebellum', 'nuchal translucency', 'neck'],
        # 'cerebellum': ['suboccipitobregmatic', 'cerebellum'],
        # 'biparietal diameter': ['biparietal diameter'],
        'face': ['face', 'nose', 'lips', 'nostrils', 'mouth', 'chin', 'forehead'],
        'cardiac': ['cardiac', 'heart'],
        'abdomen': ['abdomen', 'belly', 'stomach', 'baby\'s tummy', 'the tummy'],
        'upper limb': ['arm', 'upper limb'],
        'hand': ['hand', 'finger'],
        'lower limb': ['lower limb', 'femur', 'leg', 'thigh'],
        'foot': ['foot', 'feet', 'toes'],
        'placenta': ['placenta'],
        'kidney': ['kidney', 'renal pelvis', 'renal arteries'],
        'spine': ['spine'],
        # 'bladder': ['bladder'],
        'cord insertion': ['cord insertion'],
        # 'neck': ['nuchal translucency', 'neck']
    }

def get_VQAdataset_test(args, json_path, open):
    img_root_dir = '/media/engs2527/2TBFast/sonomate/vqa_dataset/features/'
    val_dataset = Binary_VQA_Dataset_test(json_path, img_root_dir, context_length=args.context_length, is_blank=args.is_blank, is_knowledge=args.is_knowledge)

    val_sampler = data.SequentialSampler(val_dataset) 

    val_loader = DataLoaderBG(val_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False,
        shuffle=(val_sampler is None), sampler=val_sampler, 
    )
    return val_dataset, val_loader

def get_generated_texts(label,outputs,tokenizer):
    #1,256
    # print(outputs.shape, label.shape, outputs[label!=0].shape)
    # print(label!=0, (label!=0).shape)
    # outputs = outputs[label!=0][1:-1]
    generated_text = []
    for out in outputs:
        generated_text.append(tokenizer.decoder(out))
    return generated_text

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str.lower(), target_str.lower())
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index

def accuracy(pred, label):
    acc = (pred == label)
    return int(acc)

def F1_score(pred, label):
    anatomy_list = list(anatomy.keys())
    pred_np = np.zeros(len(anatomy_list))
    label_np = np.zeros(len(anatomy_list))
    # import pdb;pdb.set_trace()
    for i in range(len(anatomy_list)):
        if anatomy_list[i] in pred:
            pred_np[i] = 1
        if anatomy_list[i] in label:
            label_np[i] = 1
    # print(label_np, pred_np)
    return f1_score(label_np, pred_np, average=None)[-1]

def edit_distance(pred, label):
    str1 = pred.replace(',', '').split(' ')
    while '' in str1:
        str1.remove('')
    str2 = label.replace(',', '').split(' ')
    while '' in str2:
        str2.remove('')
    a = len(str1)
    b = len(str2)
    string_matrix = [[0 for i in range(b+1)] for i in range(a+1)]
    for i in range(a+1):
        for j in range(b+1):
            if i == 0:
                string_matrix[i][j] = j   # If first string is empty, insert all characters of second string into first.
            elif j == 0:
                string_matrix[i][j] = i   # If second string is empty, remove all characters of first string.
            elif str1[i-1] == str2[j-1]:
                string_matrix[i][j] = string_matrix[i-1][j-1]  # If last characters of two strings are same, nothing much to do. Ignore the last two characters and get the count of remaining strings.
            else:
                string_matrix[i][j] = 1 + min(string_matrix[i][j-1],      # insert operation
                                       string_matrix[i-1][j],      # remove operation
                                       string_matrix[i-1][j-1])    # replace operation
    return string_matrix[a][b]


def bleu_score(pred, label, bleu_num=2):
    pred1 = pred.replace(',', '').split(' ')
    while '' in pred1:
        pred1.remove('')
    label1 = [label.replace(',', '').split(' ')]
    while '' in label1:
        label1.remove('')
    # import pdb;pdb.set_trace()
    # print('label: ', label1)
    # print('pred: ', pred1)
    if bleu_num == 1:
        weights = [(1.,0)]
    elif bleu_num == 2:
        weights = [(1.,0), (1./2., 1./2.)]
    elif bleu_num == 3:
        weights = [(1.,0), (1./2., 1./2.), (1./3., 1./3., 1./3.)]
    else:
        weights = [(1.,0), (1./2., 1./2.), (1./3., 1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.)]
    bleu = sentence_bleu(label1, pred1, weights)
    return bleu

@torch.no_grad()
def test_videoqa_pulse(model, device, args):

    test_json_path = [
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/VideoVQA_dataset/VideoQA_anatomy_test.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/VideoVQA_dataset/VideoQA_measure_test.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/VideoVQA_dataset/VideoQA_missing_test.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/VideoVQA_dataset/VideoQA_ba_test.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/videoQA_dataset4/VideoQA_skill_test.json'
            ]

    task = ['anatomy', 
            'measure', 
            'missing', 
            'ba',
            'skill'
            ]

    All_ACC = 0
    per_task_acc = {}
    task_num = 0
    for json_path, t in zip(test_json_path, task):
        if os.path.exists(json_path):
            if True:
                task_num += 1
                print('Perform VQA task of ', json_path)
                open = False
                if t == 'anatomy_open' or t == 'anatomy_openSource':
                    open = True
                _, val_loader = get_VQAdataset_test(args, json_path, open)

                model.eval()
                MED = 0
                ACC = 0
                F1 = 0
                Bleu1, Bleu2, Bleu3, Bleu4 = 0, 0, 0, 0
                cc = 0

                # for idx, input_data in enumerate(val_loader):
                for idx, input_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                    feat = input_data["feat"].to(device, non_blocking=True).to(dtype=torch.float32)
                    Qtoken = input_data['Question'].to(device, non_blocking=True).squeeze()
                    Answer = input_data['Answer']#.to(device, non_blocking=True)

                    B, T, _ = feat.shape

                    # Qtoken = model.tokenizer(Question, context_length=77).to(device)
                    Qfeat, _ = model.lang_model(Qtoken, if_token=True)
                    outputs = model(feat, Qfeat, VQA=True)
                    del feat, Qfeat
                    
                    outputs = outputs.argmax(-1)

                    for out in range(outputs.shape[0]):
                        out_toks = outputs[out][34:]
                        if 3 in out_toks:
                            eos_tok_ind = (out_toks == 3).nonzero(as_tuple=True)[0][0]
                            generated_text = model.tokenizer.decoder(out_toks[:eos_tok_ind.item()])
                        else:
                            generated_text = model.tokenizer.decoder(out_toks)
                        generated_text = ' '.join(generated_text).strip().replace(' ,', ',')

                        index_label = Answer[out].lower().strip()
                        index_pred = generated_text.lower().strip()
                        # if t == 'ba' and index_label == index_pred:
                        #     print(index_label, index_pred)

                        if t == 'anatomy' or t == 'measure':
                            index_pred = index_pred.replace('rum ##p', 'rump')
                            MED += edit_distance(index_pred, index_label)
                            bleu = bleu_score(index_pred, index_label)
                            Bleu1 += bleu[0]
                            Bleu2 += bleu[1]
                        if t == 'missing':
                            Bleu1 += bleu_score(index_pred, index_label, bleu_num=1)
                            F1 += F1_score(index_pred, index_label)
                        if t == 'ba' or t == 'skill':
                            Bleu1 += bleu_score(index_pred, index_label, bleu_num=1)
                            ACC += accuracy(index_pred, index_label)
                        # MED += edit_distance(index_pred, index_label)
                        # F1 += F1_score(index_pred, index_label)
                        # ACC += accuracy(index_pred, index_label)
                        # bleu = bleu_score(index_pred, index_label)
                        # Bleu1 += bleu[0]
                        # Bleu2 += bleu[1]
                        # Bleu3 += bleu[2]
                        # Bleu4 += bleu[3]
                        cc = cc + 1
                    # import pdb;pdb.set_trace()
                    # print('label: ', index_label)
                    # print('pred: ', index_pred, index_label==index_pred)
                    del outputs, index_pred

                if t == 'anatomy' or t == 'measure':
                    print(json_path, ' MED: ', MED/cc, ' Bleu1: ', Bleu1/cc, ' Bleu2: ', Bleu2/cc)
                    per_task_acc[t+'_MED'] = MED/cc
                    per_task_acc[t+'_bleu-1'] = Bleu1/cc
                    per_task_acc[t+'_bleu-2'] = Bleu2/cc
                if t == 'missing':
                    print(json_path, ' F1: ', F1/cc, ' Bleu1: ', Bleu1/cc)
                    per_task_acc[t+'_F1'] = F1/cc
                    per_task_acc[t+'_bleu-1'] = Bleu1/cc
                if t == 'ba' or t == 'skill':
                    print(json_path, ' ACC: ', ACC/cc, ' Bleu1: ', Bleu1/cc)
                    per_task_acc[t+'_ACC'] = ACC/cc
                    per_task_acc[t+'_bleu-1'] = Bleu1/cc

                All_ACC = All_ACC + Bleu1/cc
    print('Average bleu-1:', All_ACC / task_num)
    # xiaoqing
    return All_ACC / task_num, per_task_acc
