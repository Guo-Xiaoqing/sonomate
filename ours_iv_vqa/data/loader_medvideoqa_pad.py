import csv
import json
import logging
import os
import re
import difflib
import sys
import torch
import random
from abc import abstractmethod
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import open_clip

interval = 10
video_len = 350

class Binary_VQA_Dataset(Dataset):
    def __init__(self, json_path, img_root_dir, context_length, is_blank=True, is_knowledge=False):
        # json_path: vqa dataset path
        # img_root_dir: /data/engs2527/pulsedata/features/
        self.is_blank = is_blank
        self.is_knowledge = False #is_knowledge
        self.root_dir = img_root_dir
        with open(json_path) as fobj:
            self.data_info = json.load(fobj)

        model_name = 'PubMedBERT_256-timm-vit_base_patch16_224'
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.tokenizer.pad_token_id=0
        self.context_length = context_length
        
    def encode_mlm(self, question_text, answer_prompt, answer_text, mask_token= '[UNK]', pad_token='[PAD]', eos_token = '[PAD]'):
        # def measure_word_len(word):
        #     token_ids = torch.nonzero(self.tokenizer(word)).size(0)
        #     return token_ids - 2
        
        qlen = torch.nonzero(self.tokenizer(question_text)).size(0)
        question_text_tokens = question_text.split(' ') + [pad_token] * (31 - qlen)
        
        alen = torch.nonzero(self.tokenizer(answer_prompt)).size(0)
        alen1 = self.context_length - 29
        alen1 = torch.nonzero(self.tokenizer(answer_text)).size(0)
        answer_text_tokens = answer_prompt.split(' ') + [mask_token] * (alen1 - alen)
        # bert_input_tokens = []
        # bert_label_tokens = []  
        # for i, token in enumerate(answer_text_tokens): 
        #     word_len = measure_word_len(token)
        #     bert_input_tokens += [mask_token]*word_len
        #     bert_label_tokens += [token]

        bert_input_tokens = question_text_tokens + answer_text_tokens
        bert_label_tokens = question_text_tokens + answer_text.split(' ')
        return ' '.join(bert_input_tokens), ' '.join(bert_label_tokens)
    
    def encode_mask(self, question_text, mask_token= '[MASK]', pad_token='[PAD]', eos_token = '[PAD]'): 
        question_text_tokens = question_text.split()
        bert_input_tokens = []
        
        for i, token in enumerate(question_text_tokens):
            if random.random() > 0.6:
                bert_input_tokens += [mask_token]
            else:
                bert_input_tokens += [token]
        # bert_input_tokens += [eos_token]
        bert_input = ' '.join(bert_input_tokens)
        bert_label = question_text
        return bert_input

    def token_mask(self, question_token, mask_rate=0.3): 
        mask = (torch.rand(question_token.shape) > mask_rate).type(torch.FloatTensor)
        question_token_mask = question_token * mask + (torch.ones_like(mask) - mask) * 4
        question_token_mask = question_token_mask.type(torch.IntTensor)
        question_token_mask = torch.where(question_token==0, question_token, question_token_mask)
        question_token_mask = torch.where(question_token==2, question_token, question_token_mask)
        question_token_mask = torch.where(question_token==3, question_token, question_token_mask)
        return question_token_mask

    def encode_mlmK(self, K, mask_token= '[MASK]', pad_token='[PAD]', eos_token = '[PAD]'):
        def measure_word_len(word):
            return torch.nonzero(self.tokenizer(word)).size(0)
        
        K = K.split()
        if K == []:
            length = 0
        else:
            length = measure_word_len(K)
        Kbert_input_tokens = [mask_token] * length
        Kbert_label_tokens = K 

        return 'The reason is ' + ' '.join(Kbert_input_tokens), 'The reason is ' + ' '.join(Kbert_label_tokens)
        
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        # import pdb;pdb.set_trace()
        vqa = self.data_info[index]
        feat_path = os.path.join(self.root_dir, vqa["feature_id"] + '.npy')
        feat_start = int(vqa["start_time"])
        feat_end = int(vqa["end_time"])
        feat = np.load(feat_path)
        feat = feat[feat_start:feat_end:interval,:]
        Feature = np.zeros((video_len, feat.shape[1]))
        # if feat.shape[0]>video_len:
        #     Feature[:, :] = feat[:video_len,:]
        # else:
        #     Feature[:feat.shape[0], :] = feat

        question = vqa["question"].lower()
        answer = vqa["answer"].lower()
        # answer_neg = vqa["answer_neg"].lower()

        question_text = question 
        question_text = question_text#.replace('\n', '')
        answer_prompt = '[SEP] The Answer is '  
        answer_text = '[SEP] The Answer is ' + answer

        bert_input, bert_label = self.encode_mlm(question_text, answer_prompt, answer_text)
        bert_input = self.tokenizer(bert_input, context_length=self.context_length)
        bert_label = self.tokenizer(bert_label, context_length=self.context_length)

        # neg = self.tokenizer(answer_neg, context_length=20)
        # neg = torch.where(neg<5, torch.zeros_like(neg), neg)
        # neg = torch.unique(neg)[1:]
        # bert_label_neg = torch.ones((14, self.context_length))
        # for n in range(len(neg)):
        #     bert_label_neg[n] = torch.ones((1, self.context_length)) * neg[n]

        # print('input: ', bert_input, bert_input[0,29:])
        # print('label: ', bert_label, bert_label[0,29:])
        # print('label_neg: ', neg, bert_label_neg, bert_label_neg.shape)
        # import pdb;pdb.set_trace()

        qinput = torch.where(bert_input != torch.ones_like(bert_input), bert_input, torch.zeros_like(bert_input))
        qinput_mask = self.token_mask(qinput, mask_rate = 0.5)

        if self.is_knowledge:
            if "knowledge" in vqa.keys():
                knowledge = vqa["knowledge"]
            else:
                knowledge = ''
            Kbert_input, Kbert_label = self.encode_mlmK(knowledge)
            # print(Kbert_input)
            # print(Kbert_label)
            Kbert_input = self.tokenizer(Kbert_input, context_length=80)
            Kbert_label = self.tokenizer(Kbert_label, context_length=80)
            bert_input = torch.cat((bert_input, Kbert_input), 1)
            bert_label = torch.cat((bert_label, Kbert_label), 1)
            # print(bert_input, Kbert_input, bert_label, Kbert_label)[0]

            return {
                "feat": Feature,
                "feat_path": feat_path,
                'Question_orig': qinput_mask,
                'Label_orig': qinput,
                'Question': bert_input,
                'Label': bert_label,
                # 'Label_neg': bert_label_neg,
                }
        else:
            return {
                "feat": Feature,
                # "encoded_input_ids": encoded_input['input_ids'],
                # "encoded_attention_mask": encoded_input['attention_mask'],
                # "label": encoded_label['input_ids'],
                "feat_path": feat_path,
                'Question_orig': qinput_mask,
                'Label_orig': qinput,
                'Question': bert_input,
                'Label': bert_label,
                # 'Choice': choice,
                # 'Answer': answer
                # 'Label_neg': bert_label_neg,
                }
        

class Binary_VQA_Dataset_test(Dataset):
    def __init__(self, json_path, img_root_dir, context_length, is_blank=True, is_knowledge=False):
        # json_path: vqa dataset path
        # img_root_dir: /data/engs2527/pulsedata/
        self.is_blank = is_blank
        self.is_knowledge = False #is_knowledge
        self.root_dir = img_root_dir
        with open(json_path) as fobj:
            self.data_info = json.load(fobj)

        model_name = 'PubMedBERT_256-timm-vit_base_patch16_224'
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.tokenizer.pad_token_id=0
        self.context_length = context_length
        
    def encode_mlm(self, question_text, answer_prompt, answer_text, mask_token= '[UNK]', pad_token='[PAD]', eos_token = '[PAD]'):        
        qlen = torch.nonzero(self.tokenizer(question_text)).size(0)
        question_text_tokens = question_text.split(' ') + [pad_token] * (31 - qlen)
        
        alen = torch.nonzero(self.tokenizer(answer_prompt)).size(0)
        alen1 = self.context_length - 29
        alen1 = torch.nonzero(self.tokenizer(answer_text)).size(0)
        answer_text_tokens = answer_prompt.split(' ') + [mask_token] * (alen1 - alen)

        bert_input_tokens = question_text_tokens + answer_text_tokens
        bert_label_tokens = question_text_tokens + answer_text.split(' ')
        return ' '.join(bert_input_tokens), ' '.join(bert_label_tokens)
    
    def encode_mlmK(self, K, mask_token= '[MASK]', pad_token='[PAD]', eos_token = '[PAD]'):
        def measure_word_len(word):
            return torch.nonzero(self.tokenizer(word)).size(0)
        
        K = K.split()
        if K == []:
            length = 0
        else:
            length = measure_word_len(K)
        Kbert_input_tokens = [mask_token] * length
        Kbert_label_tokens = K 

        return 'The reason is ' + ' '.join(Kbert_input_tokens), 'The reason is ' + ' '.join(Kbert_label_tokens)
             
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        vqa = self.data_info[index]
        feat_path = os.path.join(self.root_dir, vqa["feature_id"] + '.npy')
        feat_start = int(vqa["start_time"])
        feat_end = int(vqa["end_time"])
        feat = np.load(feat_path)
        feat = feat[feat_start:feat_end:interval,:]
        Feature = np.zeros((video_len, feat.shape[1]))
        # if feat.shape[0]>video_len:
        #     Feature[:, :] = feat[:video_len,:]
        # else:
        #     Feature[:feat.shape[0], :] = feat

        question = vqa["question"].lower()
        answer = vqa["answer"].lower()

        question_text = question #+ ' [SEP] The Answer is '
        # question_text = question_text.replace('\n', '')
        answer_prompt = '[SEP] The Answer is ' #+ answer
        answer_text = '[SEP] The Answer is ' + answer
        bert_input, _ = self.encode_mlm(question_text, answer_prompt, answer_text)
        bert_input = self.tokenizer(bert_input, context_length=self.context_length)
        
        if self.is_knowledge:
            if "knowledge" in vqa.keys():
                knowledge = vqa["knowledge"]
            else:
                knowledge = ''
            Kbert_input, Kbert_label = self.encode_mlmK(knowledge)
            Kbert_input = self.tokenizer(Kbert_input, context_length=80)
            # Kbert_label = self.tokenizer(Kbert_label, context_length=80)
            bert_input = torch.cat((bert_input, Kbert_input), 1)
            # bert_label = torch.cat((bert_label, Kbert_label), 1)
 
        return {
            "name": feat_path,
            "feat": Feature,
            # "encoded_input_ids": encoded_input['input_ids'],
            # "encoded_attention_mask": encoded_input['attention_mask'],
            # "label": encoded_label['input_ids'],
            # "feat_path": feat_path,
            'Question': bert_input,
            'Answer': answer
            # 'knowledge': knowledge
            }
        
