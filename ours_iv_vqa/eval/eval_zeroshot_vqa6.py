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
from data.loader_medvqa6 import Binary_IVQA_Dataset, Binary_IVQA_Dataset_test
import difflib 
import random 

def get_VQAdataset_test(args, json_path):
    img_root_dir = '/media/engs2527/2TBFast/sonomate/vqa_dataset/'
    val_dataset = Binary_IVQA_Dataset_test(json_path, img_root_dir, context_length=args.context_length, is_blank=args.is_blank, is_knowledge=args.is_knowledge)

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

@torch.no_grad()
def test_vqa_pulse(model, device, args):

    is_blank = args.is_blank

    if True: #is_blank:
        test_json_path = [
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_measurement.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505_orig/testImageQA_trim.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_anatomy.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_anatomy_open.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_anatomy_openSource.json'
            ]
    else:
        test_json_path = [
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_measurement.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_trim.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_anatomy.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_anatomy_open.json',
            '/media/engs2527/2TBFast/sonomate/vqa_dataset/newdata20231111/ImageVQA_dataset20240505/testImageQA_anatomy_openSource.json'
            ]
    task = [
            'measurement', 
            'trim', 
            'anatomy', 
            'anatomy_open', 
            'anatomy_openSource'
            ]

    All_ACC = 0
    per_task_acc = {}
    task_num = 0

    for json_path, t in zip(test_json_path, task):
        if os.path.exists(json_path):
            # if t == 'freeze' or t == 'trim':
            #     rand_num = 0.0
            # else:
            #     rand_num = 0.
            # if random.random() > rand_num or (t == 'anatomy_open' and task_num == 0):
            if True:
                task_num += 1
                print('Perform VQA task of ', json_path)
                _, val_loader = get_VQAdataset_test(args, json_path)

                model.eval()
                ACC = 0
                cc = 0
                
                # for idx, input_data in enumerate(val_loader):
                for idx, input_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                    feat = input_data["feat"].to(device, non_blocking=True).to(dtype=torch.float32)
                    Qtoken = input_data['Question'].to(device, non_blocking=True).squeeze()
                    # Ltoken = input_data['Label']#.to(device, non_blocking=True)
                    Choices = input_data['Choice']#.to(device, non_blocking=True)
                    Choice = input_data['choiceABC']#.to(device, non_blocking=True)
                    Answer = input_data['Answer']#.to(device, non_blocking=True)
                    Label = input_data["Label"]
                    name = input_data["name"]
                    

                    B, _ = feat.shape

                    # Qtoken = model.tokenizer(Question, context_length=77).to(device)
                    Qfeat, _ = model.lang_model(Qtoken, if_token=True)
                    outputs = model(feat, Qfeat, VQA=True)
                    del feat, Qfeat
                    
                    outputs = outputs.argmax(-1)
                    outputs1 = outputs.clone() 
                    outputs = torch.where(Qtoken == torch.ones_like(Qtoken), outputs, torch.zeros_like(Qtoken))

                    # Atoken = model.tokenizer(Answer, context_length=77).to(device)
                    # generated_texts = get_generated_texts(Atoken, outputs, model.tokenizer)

                    for out in range(outputs.shape[0]):
                        generated_text = model.tokenizer.decoder(outputs[out])
                        generated_text = ' '.join(generated_text).strip() 

                        if is_blank:
                            index_label = Label[out].lower().strip()
                            index_pred = generated_text.lower().strip()
                               
                        else:
                            cho = Choices[out][2:].replace(' a ', ' ').replace(' b ', ' ').replace(' c ', ' ').replace(' d ', ' ').split(' [SEP] ')
                            index_label = find_most_similar_index(cho, Label[out].lower().strip())
                            index_pred = find_most_similar_index(cho, generated_text.lower().strip())
                            # index_label = Answer[out].lower().strip()
                            # index_pred = generated_text.lower().strip()
                            # print(Choice[out], Answer[out].lower(), generated_text.lower())
                
                        if index_pred == index_label:
                            ACC = ACC +1
                        cc = cc + 1
                    
                    del outputs, index_pred, outputs1
                print(json_path, ' VQA accuracy: ', ACC/cc)
                per_task_acc[t] = ACC/cc
                All_ACC = All_ACC + ACC/cc
    print('Average accuracy:', All_ACC / task_num)
    return All_ACC / task_num, per_task_acc
