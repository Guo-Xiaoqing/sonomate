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
from sklearn.metrics import recall_score
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
#             'NOSE_LIPS': 'nose or lip', 'SPINE_COR': 'spine', 'SPINE_SAG': 'spine'}
# label_set = ['heart', 'abdomen', 'femur', 'head', 'cerebellum', 'kidney', 'nose or lip', 'spine']
# name_corr_num = {'heart':0, 'abdomen':1, 'femur':2, 'head': 3, 'cerebellum':4, 'kidney':5, 'nose or lip': 6, 'spine': 7}
# class_dist = {'heart':614, 'abdomen':161, 'femur':175, 'head': 142, 'cerebellum':125, 'kidney':162, 'nose or lip': 187, 'spine': 466, 'cord insertion': 0, 'feet': 0, 'placenta': 0}

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
label_set = ['cardiac', 'abdominal circumference plane', 'femur length plane', 'head circumference plane', 'cerebellum', 'kidney', 'nose or lip', 'spine']#, 'cord insertion', 'feet', 'placenta']
name_corr_num = {'cardiac':0, 'abdominal circumference plane':1, 'femur length plane':2, 'head circumference plane': 3, 'cerebellum':4, 'kidney':5, 'nose or lip': 6, 'spine': 7}#, 'cord insertion': 8, 'feet': 9, 'placenta': 10}
class_dist = {'cardiac':614, 'abdominal circumference plane':161, 'femur length plane':175, 'head circumference plane': 142, 'cerebellum':125, 'kidney':162, 'nose or lip': 187, 'spine': 466, 'cord insertion': 0, 'feet': 0, 'placenta': 0}


class_num = 8
label_set_abb = ['Cardiac', 'ACP', 'FLP', 'HCP', 'SOB', 'Kidney', 'Nose_lips', 'Spine']
label_set_complex = []
import json
json_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/anatomy_graph.json'
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

feature_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/val_2nd_class/Second_Trimester_Classfication_Dataset/'
# label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/val_2nd_class/Second_Trimester_Classfication_Dataset.txt'
label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/val_2nd_class/Second_Trimester_Classfication_Dataset_new.txt'

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
def test_2ndclass_pulse_val(model, device, args):
    print('Perform pulsedataset 2nd val trimester classification (contain 2032 images and 8 classes)')
    model.eval()
    batch_size=200

    D = Pulse_2ndclass_data()
    data_loader = DataLoaderFast(D, batch_size=batch_size, num_workers=0)

    template = 'this is a second trimester ultrasound scan of '
    # token = model.tokenizer([template + l for l in label_set], context_length=50).to(device)

    class_template = [template + l for l in label_set]
    class_template_complex = [template + l for l in label_set_complex]
    class_template.extend(class_template_complex)

    token = model.tokenizer(class_template, context_length=args.context_length).to(device)
    text_features, logit_scale = model.lang_model(token)

    Acc_score = 0
    test_img_len = 0
    Acc_perclass_score = np.zeros(class_num)
    test_perclass_len = np.zeros(class_num)
    CM = 0
    for input_data in tqdm(data_loader, total=len(data_loader)):
        image_features  = input_data['feat'].to(device)
        labels  = input_data['label']
        names  = input_data['name']

        assert text_features.shape[0] == class_num * 2 
        logits = (logit_scale * image_features @ text_features[:class_num].t()).detach().softmax(dim=-1)
        logits += (logit_scale * image_features @ text_features[class_num:].t()).detach().softmax(dim=-1)
        predict = torch.argmax(logits, dim=-1).cpu().numpy()

        acc = np.sum(predict == np.array(labels)) #/ predict.shape[0] 
        Acc_score += acc
        test_img_len += predict.shape[0] 

        acc_perclass = recall_score(np.array(labels), predict, labels = np.array(label_set_num), zero_division=0, average=None)
        l = np.array(labels)
        l_account = np.zeros((l.size, class_num))
        l_account[np.arange(l.size), l] = 1
        l_account = l_account.sum(0)
        Acc_perclass_score += acc_perclass * l_account
        test_perclass_len += l_account

        # CM = CM + confusion_matrix(np.array(labels), predict, labels = label_set_num)

        del image_features
    accuracy = Acc_score/test_img_len
    accuracy_perclass = Acc_perclass_score/test_perclass_len
    print('Classification accuracy: ', accuracy, 'Perclass accuracy: ', accuracy_perclass)

    # plot_confusion_matrix(CM, label_set_num) 
    return accuracy#, accuracy_perclass

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
    plt.savefig('./vis/CM_vis_2ndval_norm.png')
    plt.cla()
    plt.clf()
    plt.close('all')   
    # gc.collect()
    # xiaoqing

######### BiomedCLIP val classification accuracy: 0.2421259842519685 (11 classes)
######### BiomedCLIP val classification accuracy: 0.3494094488188976 (8 classes)

## Classification accuracy:  0.23179133858267717 (name more complex) 
## Perclass accuracy:  [0.09609121 0.0310559  0.86857143 0.23239437 0.904      0.00617284 0.16042781 0.16738197]

## BiomedCLIP test Classification accuracy:  0.3395669291338583 (with second trimester in prompt) 
## Perclass accuracy:  [0.44136808 0.         0.56       0.13380282 0.888      0.   0.58823529 0.17381974]

## BiomedCLIP test Classification accuracy:  0.4360236220472441 (with second trimester in prompt with anatomy graph) 
## Perclass accuracy:  [0.71986971 0.         0.72       0.         0.992      0.02469136 0.76470588 0.10085837]

## label_set = ['cardiac', 'abdominal circumference plane', 'femur length plane', 'head circumference plane', 'cerebellum', 'kidney', 'nose or lip', 'spine']