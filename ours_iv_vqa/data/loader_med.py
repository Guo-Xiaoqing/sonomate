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


class Binary_Dataset(Dataset):
    def __init__(self, json_path, img_root_dir, context_length):
        # json_path: vqa dataset path
        # img_root_dir: /data/engs2527/pulsedata/features/
        self.root_dir = img_root_dir
        with open(json_path) as fobj:
            self.data_info = json.load(fobj)

        model_name = 'PubMedBERT_256-timm-vit_base_patch16_224'
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.tokenizer.pad_token_id=0
        self.context_length = context_length

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
        interval = 2
        video_len = 350
        vqa = self.data_info[index]
        feat_path = os.path.join(self.root_dir, vqa["feature_id"].split('_')[0] + '.npy')
        feat_start = int(vqa["start_time"])
        feat_end = int(vqa["end_time"])
        feat = np.load(feat_path)
        feat = feat[feat_start:feat_end:interval,:]
        Feature = np.zeros((video_len, feat.shape[1]))
        if feat.shape[0]>video_len:
            Feature[:, :] = feat[:video_len,:]
        else:
            Feature[:feat.shape[0], :] = feat

        qinput = vqa["sentence"].lower()
        qinput = self.tokenizer(qinput, context_length=self.context_length)
        qinput = torch.where(qinput != torch.ones_like(qinput), qinput, torch.zeros_like(qinput))
        qinput_mask = self.token_mask(qinput, mask_rate = 0.5)

        return {
                "feat": Feature,
                "feat_path": feat_path,
                'qinput': qinput,
                'qinput_mask': qinput_mask,
                }
        
    def __len__(self):
        return len(self.data_info)
    
    
