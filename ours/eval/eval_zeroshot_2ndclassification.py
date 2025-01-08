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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content

# name_corr = {'3VT':'heart', '3VV':'heart', '4CH':'heart', 'RVOT':'heart', 'LVOT':'heart', 'ACP':'abdomen', 'FLP':'femur', 
#             'HCP': 'head', 'SOB':'cerebellum', 'KIDNEY':'kidney', 'KIDNEYS': 'kidney', 'KIDNEYS_CORONAL': 'kidney',
#             'NOSE_LIPS': 'nose or lip', 'SPINE_COR': 'spine', 'SPINE_SAG': 'spine', 
#             'CORD_INSERT': 'cord insertion', 'FEET': 'feet', 'PLAC_ANTE': 'placenta', 'PLAC_POST': 'placenta'}
# label_set = ['heart', 'abdomen', 'femur', 'head', 'cerebellum', 'kidney', 'nose or lip', 'spine', 'cord insertion', 'feet', 'placenta']
# name_corr_num = {'heart':0, 'abdomen':1, 'femur':2, 'head': 3, 'cerebellum':4, 'kidney':5, 'nose or lip': 6, 'spine': 7, 'cord insertion': 8, 'feet': 9, 'placenta': 10}
# class_dist = {'heart':1542, 'abdomen':574, 'femur':540, 'head': 526, 'cerebellum':467, 'kidney':523, 'nose or lip': 520, 'spine': 1014, 'cord insertion': 114, 'feet': 64, 'placenta': 73}

# name_corr = {
#     '3VT':'cardiac', 
#     '3VV':'cardiac', 
#     '4CH':'cardiac', 
#     'RVOT':'cardiac', 
#     'LVOT':'cardiac', 
#     'ACP':'abdominal circumference plane', 
#     'FLP':'femur length plane', 
#     'HCP': 'head circumference plane', 
#     'SOB':'cerebellum', 
#     'KIDNEY':'kidney', 
#     'KIDNEYS': 'kidney', 
#     'KIDNEYS_CORONAL': 'kidney',
#     'NOSE_LIPS': 'nose or lip', 
#     'SPINE_COR': 'spine', 
#     'SPINE_SAG': 'spine', 
#     'CORD_INSERT': 'cord insertion', 
#     'FEET': 'feet', 
#     'PLAC_ANTE': 'placenta', 
#     'PLAC_POST': 'placenta'
#     }
# label_set = ['cardiac', 'abdominal circumference plane', 'femur length plane', 'head circumference plane', 'cerebellum', 'kidney', 'nose or lip', 'spine', 'cord insertion', 'feet', 'placenta']
# name_corr_num = {'cardiac':0, 'abdominal circumference plane':1, 'femur length plane':2, 'head circumference plane': 3, 'cerebellum':4, 'kidney':5, 'nose or lip': 6, 'spine': 7, 'cord insertion': 8, 'feet': 9, 'placenta': 10}
# class_dist = {'cardiac':1542, 'abdominal circumference plane':574, 'femur length plane':540, 'head circumference plane': 526, 'cerebellum':467, 'kidney':523, 'nose or lip': 520, 'spine': 1014, 'cord insertion': 114, 'feet': 64, 'placenta': 73}
# class_num = 11
# label_set_abb = ['Cardiac', 'ACP', 'FLP', 'HCP', 'SOB', 'Kidney', 'Nose_lips', 'Spine', 'CORD_INSERT', 'FEET', 'PLA']


name_corr = {
    '3VT':'cardiac', 
    '3VV':'cardiac', 
    '4CH':'cardiac', 
    'RVOT':'cardiac', 
    'LVOT':'cardiac', 
    'ACP':'abdominal circumference plane', 
    'FLP':'femur length plane', 
    'HCP': 'head circumference plane', 
    'SOB':'cerebellum', 
    'KIDNEY':'kidney', 
    'KIDNEYS': 'kidney', 
    'KIDNEYS_CORONAL': 'kidney',
    'NOSE_LIPS': 'nose or lip', 
    'SPINE_COR': 'spine', 
    'SPINE_SAG': 'spine'
    }
label_set = ['cardiac', 'abdominal circumference plane', 'femur length plane', 'head circumference plane', 'cerebellum', 'kidney', 'nose or lip', 'spine']
name_corr_num = {'cardiac':0, 'abdominal circumference plane':1, 'femur length plane':2, 'head circumference plane': 3, 'cerebellum':4, 'kidney':5, 'nose or lip': 6, 'spine': 7}
class_dist = {'cardiac':1542, 'abdominal circumference plane':574, 'femur length plane':540, 'head circumference plane': 526, 'cerebellum':467, 'kidney':523, 'nose or lip': 520, 'spine': 1014}
class_num = 8
label_set_abb = ['Cardiac', 'ACP', 'FLP', 'HCP', 'SOB', 'Kidney', 'Nose_lips', 'Spine']


label_set_complex = []
import json
json_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/anatomy_graph_new.json'
with open(json_path) as fobj:
    anatomy_graph = json.load(fobj)

    for abb in label_set_abb:
        subcategory = anatomy_graph[abb]["subcategory"]
        abb_sentence = ''
        for sub in subcategory:
            abb_sentence += sub + ' '
        label_set_complex.append(abb_sentence)


label_set_num = list(range(class_num))
name_corr_keys = list(name_corr.keys())

# os.environ['CUDA_VISIBLE_DEVICES']='0'

feature_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_2nd_class/Second_Trimester_Classfication_Dataset/'
# label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_2nd_class/Second_Trimester_Classfication_Dataset.txt'
label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_2nd_class/Second_Trimester_Classfication_Dataset_8classes.txt'

class Pulse_2ndclass_data():
    """Pulse 2nd trimester classification dataset. 
    For each image, return all the visual features and all the classification labels."""
    def __init__(self):
        self.video_feature_path = feature_path
        self.anno = open(label_path, 'r').readlines()

        self.files = []
        for i in self.anno:
            feat = i.split('--')[0]
            label = i.split('--')[1]#.split('\n')[0]
            self.files.append({
                "feat": feat,
                "label": label
            })
            assert os.path.exists(os.path.join(self.video_feature_path, feat))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        datafiles = self.files[idx]
        name = datafiles["feat"]
        feature = np.load(os.path.join(self.video_feature_path, name))
        label = name_corr_num[datafiles["label"]]
        return {'feat': torch.tensor(feature.copy()),
                'label': label,
                'name': name}
                
@torch.no_grad()
def test_2ndclass_pulse(model, device, args):
    # print('Perform pulsedataset 2nd trimester classification (contain 5476 images and 11 classes)')
    print('Perform pulsedataset 2nd trimester classification (contain 5225 images and 8 classes)')
    model.eval()
    batch_size=200

    D = Pulse_2ndclass_data()
    data_loader = DataLoaderFast(D, batch_size=batch_size, num_workers=0)

    template = 'a second trimester ultrasound scan containing '
    # template = 'this is a second trimester ultrasound scan of '
    # token = model.tokenizer([template + l for l in label_set], context_length=50).to(device)

    class_template = [template + l for l in label_set]
    class_template_complex = [template + l for l in label_set_complex]
    class_template.extend(class_template_complex)

    # token = model.tokenizer(class_template, context_length=args.context_length).to(device)
    token = model.tokenizer(class_template, context_length=32).to(device)
    text_features, logit_scale = model.lang_model(token)

    cm = 0
    Acc_perclass_score = np.zeros(class_num)
    test_perclass_len = np.zeros(class_num)
    for input_data in tqdm(data_loader, total=len(data_loader)):
        image_features  = input_data['feat'].to(device)
        labels  = input_data['label']
        names  = input_data['name']
        image_features = image_features + model.img_proj_clip(image_features)

        assert text_features.shape[0] == class_num * 2 
        logits = (logit_scale * image_features @ text_features[:class_num].t()).detach().softmax(dim=-1)
        logits += (logit_scale * image_features @ text_features[class_num:].t()).detach().softmax(dim=-1)
        predict = torch.argmax(logits, dim=-1).cpu().numpy()

        cm += confusion_matrix(labels.cpu().numpy(), predict, labels = np.array(label_set_num))

        acc_perclass = recall_score(labels.cpu().numpy(), predict, labels = np.array(label_set_num), zero_division=0, average=None)
        l = labels.cpu().numpy()
        l_account = np.zeros((l.size, class_num))
        l_account[np.arange(l.size), l] = 1
        l_account = l_account.sum(0)
        Acc_perclass_score += acc_perclass * l_account
        test_perclass_len += l_account

        del image_features

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    recall = TP / (TP+FN+10e-10)
    precision = TP / (TP+FP+10e-10)
    f1 = 2*TP/(2*TP + FP + FN+10e-10)

    print('recall: ', recall.mean(), ' precision: ', precision.mean(), ' f1: ', f1.mean(), 'perclass recall: ', Acc_perclass_score/test_perclass_len)

    # plot_confusion_matrix(CM, label_set_num) 
    return f1.mean()#, accuracy_perclass

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.binary):
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(np.float32(cm)/np.sum(cm, axis=1, keepdims=True), interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./vis/CM_vis_2nd_norm.png')
    plt.cla()
    plt.clf()
    plt.close('all')   
    # gc.collect()
    # xiaoqing

########### BiomedCLIP test classification accuracy (11 classes)

## BiomedCLIP test Classification accuracy:  0.2070854638422206 (name more complex) 
## Perclass accuracy:  [0.07198444 0.01875    0.75892857 0.20634921 0.93164557 0.002457 0.05780347 0.12990937 0.39473684 0.046875   0.09589041]

## BiomedCLIP test Classification accuracy:  0.23173849525200876 (with second trimester in prompt) 
## Perclass accuracy:  [0.32230869 0.00208333 0.35044643 0.16099773 0.86075949 0.004914 0.13487476 0.07250755 0.         0.90625    0.01369863]

## BiomedCLIP test Classification accuracy:  0.30204528853177504 (with second trimester in prompt with anatomy graph) 
## Perclass accuracy:  [0.63424125 0.00208333 0.29464286 0.         0.98481013 0.00982801 0.11560694 0.02416918 0.0877193  0.859375   0.01369863]

## label_set = ['cardiac', 'abdominal circumference plane', 'femur length plane', 'head circumference plane', 'cerebellum', 'kidney', 'nose or lip', 'spine', 'cord insertion', 'feet', 'placenta']

########### BiomedCLIP test classification accuracy (8 classes)

## BiomedCLIP test Classification accuracy:  0.31311004784688995 (with second trimester in prompt) 
## Perclass accuracy:  [0.33463035 0.00208333 0.55580357 0.16099773 0.86075949 0.004914 0.58574181 0.15407855]

## BiomedCLIP test Classification accuracy:  0.4796172248803828 (with second trimester in prompt with anatomy graph) 
## Classification accuracy:  0.4796172248803828 Perclass accuracy:  [0.87937743 0.         0.33928571 0.1814059  0.77721519 0.00737101 0.62235067 0.28700906]

## label_set = ['cardiac', 'abdominal circumference plane', 'femur length plane', 'head circumference plane', 'cerebellum', 'kidney', 'nose or lip', 'spine']
