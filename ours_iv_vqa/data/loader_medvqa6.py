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


class Binary_IVQA_Dataset(Dataset):
    def __init__(self, json_path, img_root_dir, context_length, is_blank=True, is_knowledge=False):
        # json_path: vqa dataset path
        # img_root_dir: /data/engs2527/pulsedata/features/
        self.is_blank = is_blank
        self.is_knowledge = is_knowledge
        self.root_dir = img_root_dir
        with open(json_path) as fobj:
            self.data_info = json.load(fobj)

        model_name = 'PubMedBERT_256-timm-vit_base_patch16_224'
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.tokenizer.pad_token_id=0
        self.context_length = context_length
        
    def encode_mlm(self, question_text, question_text_with_answer, mask_token= '[UNK]', pad_token='[PAD]', eos_token = '[PAD]'):
        def measure_word_len(word):
            token_ids = self.tokenizer(word)
            return len(token_ids) - 1
        
        question_text_with_answer_tokens = question_text_with_answer.split()
        question_text_tokens = question_text.split()
        bert_input_tokens = []
        bert_label_tokens = []  
        
        for i, token in enumerate(question_text_with_answer_tokens):
            if i < len(question_text_tokens):
                bert_input_tokens += [token]
                bert_label_tokens += [token]#*word_len
            else:
                bert_input_tokens += [mask_token]#*word_len
                bert_label_tokens += [token]
        # bert_input_tokens += [eos_token]
        # bert_label_tokens += [eos_token]
        bert_input = ' '.join(bert_input_tokens)
        bert_label = ' '.join(bert_label_tokens)
        return bert_input, bert_label

    def encode_mlmK(self, K, mask_token= '[MASK]', pad_token='[PAD]', eos_token = '[PAD]'):
        def measure_word_len(word):
            return torch.nonzero(self.tokenizer(word)).size(0)
        
        K = K.split()
        if K == []:
            length = 0
        else:
            length = measure_word_len(K)
        length = 50
        Kbert_input_tokens = [mask_token] * length
        Kbert_label_tokens = K 

        return 'The reason is ' + ' '.join(Kbert_input_tokens), 'The reason is ' + ' '.join(Kbert_label_tokens)
        
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

    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        vqa = self.data_info[index]
        feat_path = os.path.join(self.root_dir, vqa["feature_id"].split('_')[0] + '.npy')
        feat_ind = int(vqa["feature_id"].split('_ind')[1])
        feat = np.load(feat_path)
        if feat_ind >= feat.shape[0]:
            feat = feat[-1,:]
        else:
            feat = feat[feat_ind,:]
        question = vqa["question"].lower()
        choice = [i.lower() for i in vqa["multiple_choices"]]
        choice = ' [SEP] '.join(choice)
        # choice = choice[2:].replace(' a ', ' ').replace(' b ', ' ').replace(' c ', ' ').replace(' d ', ' ')
        answer = vqa["answer"].lower()
        label = vqa["label"].lower()
        if 'annotation' in vqa.keys() and label == 'other':
            label = vqa["annotation"].lower()

        if self.is_blank:
            # question_text = 'Question is '+ question +' The Answer is '
            question_text = question + ' [SEP] The Answer is '
            question_text = question_text.replace('\n', '')
            question_text_with_answer = question_text + label
            # if 'day' in question_text_with_answer:
            #     question_text = question + ' [SEP] ' + choice + ' [SEP] The Answer is '
            #     question_text = question_text.replace('\n', '')
            #     question_text_with_answer = question_text + label 
            # print(question_text_with_answer)
        else:
            # question_text = 'Question is '+ question + ' The choices are ' + choice + '. The Answer is '
            question_text = question + ' [SEP] ' + choice + ' [SEP] The Answer is '
            question_text = question_text.replace('\n', '')
            question_text_with_answer = question_text + label #answer #+ ' ' + label

        bert_input, bert_label = self.encode_mlm(question_text, question_text_with_answer)
        # print(bert_input)
        # print(bert_label)
        bert_input = self.tokenizer(bert_input, context_length=self.context_length)
        bert_label = self.tokenizer(bert_label, context_length=self.context_length)

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
            Kbert_input = self.tokenizer(Kbert_input, context_length=50)
            Kbert_label = self.tokenizer(Kbert_label, context_length=50)
            bert_input = torch.cat((bert_input, Kbert_input), 1)
            bert_label = torch.cat((bert_label, Kbert_label), 1)
            # print(bert_input, Kbert_input, bert_label, Kbert_label)[0]

        # feat = np.zeros_like(feat)
        return {
                "feat": feat,
                "feat_path": feat_path,
                'Question_orig': qinput_mask,
                'Label_orig': qinput,
                'Question': bert_input,
                'Label': bert_label,
                # 'KQuestion': Kbert_input,
                # 'KLabel': Kbert_label,
                }
        

class Binary_IVQA_Dataset_test(Dataset):
    def __init__(self, json_path, img_root_dir, context_length, is_blank=True, is_knowledge=False):
        # json_path: vqa dataset path
        # img_root_dir: /data/engs2527/pulsedata/
        self.is_blank = is_blank
        self.is_knowledge = is_knowledge
        self.root_dir = img_root_dir
        with open(json_path) as fobj:
            self.data_info = json.load(fobj)

        model_name = 'PubMedBERT_256-timm-vit_base_patch16_224'
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.tokenizer.pad_token_id=0
        self.context_length = context_length
        
    def encode_mlm(self, question_text, question_text_with_answer, mask_token= '[UNK]', pad_token='[PAD]', eos_token = '[PAD]'):
        def measure_word_len(word):
            token_ids = self.tokenizer(word)
            return len(token_ids) - 1
        
        question_text_with_answer_tokens = question_text_with_answer.split()
        question_text_tokens = question_text.split()
        bert_input_tokens = []
        bert_label_tokens = []  
        
        for i, token in enumerate(question_text_with_answer_tokens):
            if i < len(question_text_tokens):
                bert_input_tokens += [token]
                # bert_label_tokens += [pad_token]#*word_len
            else:
                bert_input_tokens += [mask_token]#*word_len
                # bert_label_tokens += [token]
        # bert_input_tokens += [eos_token]
        # bert_label_tokens += [eos_token]
        bert_input = ' '.join(bert_input_tokens)
        # bert_label = ' '.join(bert_label_tokens)
        return bert_input,bert_input

    def encode_mlmK(self, K, mask_token= '[MASK]', pad_token='[PAD]', eos_token = '[PAD]'):
        def measure_word_len(word):
            return torch.nonzero(self.tokenizer(word)).size(0)
        
        K = K.split()
        if K == []:
            length = 0
        else:
            length = measure_word_len(K)
        length = 50
        Kbert_input_tokens = [mask_token] * length
        Kbert_label_tokens = K 

        return 'The reason is ' + ' '.join(Kbert_input_tokens), 'The reason is ' + ' '.join(Kbert_label_tokens)
            
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        vqa = self.data_info[index]
        feat_path = os.path.join(self.root_dir, vqa["feature_id"])
        feat = np.load(feat_path)
        question = vqa["question"].lower()
        choice = [i.lower() for i in vqa["multiple_choices"]]
        choice = ' [SEP] '.join(choice)
        choiceABC = ' '.join(vqa['multiple_choicesABC']).lower()
        answer = vqa["answer"].lower()
        label = vqa["label"].lower()
        annotation = vqa["label"].lower()
        if 'annotation' in vqa.keys():
            annotation = vqa["annotation"].lower()
        if 'annotation' in vqa.keys() and label == 'other':
            label = vqa["annotation"].lower()
            choice = choice.replace('other', label)

        if self.is_blank:
            # question_text = 'Question is '+ question +' The Answer is '
            question_text = question + ' [SEP] The Answer is '
            question_text = question_text.replace('\n', '')
            question_text_with_answer = question_text + label
            # if 'day' in question_text_with_answer:
            #     question_text = question + ' [SEP] ' + choice + ' [SEP] The Answer is '
            #     question_text = question_text.replace('\n', '')
            #     question_text_with_answer = question_text + label 
        else:
            # question_text = 'Question is '+ question + ' The choices are ' + choice + '. The Answer is '
            question_text = question + ' [SEP] ' + choice + ' [SEP] The Answer is '
            question_text = question_text.replace('\n', '')
            question_text_with_answer = question_text + label #answer
        bert_input, bert_label = self.encode_mlm(question_text, question_text_with_answer)
        bert_input = self.tokenizer(bert_input, context_length=self.context_length)
        
        if self.is_knowledge:
            if "knowledge" in vqa.keys():
                knowledge = vqa["knowledge"]
            else:
                knowledge = ''
            Kbert_input, Kbert_label = self.encode_mlmK(knowledge)
            Kbert_input = self.tokenizer(Kbert_input, context_length=50)
            # Kbert_label = self.tokenizer(Kbert_label, context_length=50)
            bert_input = torch.cat((bert_input, Kbert_input), 1)
            # bert_label = torch.cat((bert_label, Kbert_label), 1)
 

        # feat = np.zeros_like(feat)
        return {
                "name": feat_path,
                "feat": feat,
                'Question': bert_input,
                'Choice': choice,
                'choiceABC': choiceABC,
                'Answer': answer,
                'Label': label,
                'annotation': annotation,
                # 'knowledge': knowledge
                }
        
