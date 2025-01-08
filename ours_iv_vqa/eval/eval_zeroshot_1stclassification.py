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

# name_corr = {'3D':'3D view', 'ABD':'abdomen', 'ASK':'placenta', 'BPD':'brain', 
#             'CRL':'sagittal view of fetus', 'NT':'nuchal translucency'}
# label_set = ['3D view', 'abdomen', 'placenta', 'brain', 'sagittal view of fetus', 'nuchal translucency']
# name_corr_num = {'3D view':0, 'abdomen':1, 'placenta':2, 'brain':3, 'sagittal view of fetus':4, 'nuchal translucency':5}
# class_dist = {'3D view':4207, 'abdomen':6992, 'placenta':2548, 'brain':6012, 'sagittal view of fetus':48243, 'nuchal translucency':15545}


name_corr = {
    '3D':'3D view', 
    'ABD':'abdomen', 
    'ASK':'placenta', 
    'BPD':'biparietal diameter', 
    'CRL':'sagittal view of fetus', 
    'NT':'nuchal translucency'
    }
# label_set = ['3D view', 'abdomen', 'placenta', 'biparietal diameter', 'sagittal view of fetus', 'nuchal translucency']
# name_corr_num = {'3D view':0, 'abdomen':1, 'placenta':2, 'biparietal diameter':3, 'sagittal view of fetus':4, 'nuchal translucency':5}
# class_dist = {'3D view':4207, 'abdomen':6992, 'placenta':2548, 'biparietal diameter':6012, 'sagittal view of fetus':48243, 'nuchal translucency':15545}
# class_num = 6
# label_set_abb = ['3D', 'ABD', 'ASK', 'BPD', 'CRL', 'NT']

# label_set = ['3D view', 'abdomen', 'placenta', 'biparietal diameter', 'nuchal translucency']
# name_corr_num = {'3D view':0, 'abdomen':1, 'placenta':2, 'biparietal diameter':3, 'nuchal translucency':4}
# class_dist = {'3D view':4207, 'abdomen':6992, 'placenta':2548, 'biparietal diameter':6012, 'nuchal translucency':15545}
# class_num = 5
# label_set_abb = ['3D', 'ABD', 'ASK', 'BPD', 'NT']

label_set = ['3D view', 'abdomen', 'placenta', 'biparietal diameter', 'sagittal view of fetus']
name_corr_num = {'3D view':0, 'abdomen':1, 'placenta':2, 'biparietal diameter':3, 'sagittal view of fetus':4}
class_dist = {'3D view':4207, 'abdomen':6992, 'placenta':2548, 'biparietal diameter':6012, 'sagittal view of fetus':5851}
class_num = 5
label_set_abb = ['3D', 'ABD', 'ASK', 'BPD', 'CRL']
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

feature_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_1st_class/First_Trimnester_Classfication_Dataset/'
# label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_1st_class/First_Trimnester_Classfication_Dataset.txt'
# label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_1st_class/First_Trimnester_Classfication_Dataset_new.txt'
# label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_1st_class/First_Trimnester_Classfication_Dataset_noCRL.txt'
label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_1st_class/First_Trimnester_Classfication_Dataset_5classes.txt'

class Pulse_1stclass_data():
    """Pulse 1st trimester classification dataset. 
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
def test_1stclass_pulse(model, device, args):
    # print('Perform pulsedataset 1st trimester classification (contain 83547 images and 6 classes)')
    print('Perform pulsedataset 1st trimester classification (contain 25610 images and 5 classes)')
    # print('Perform pulsedataset 1st trimester classification (contain 35304 images and 5 classes)')
    model.eval()
    batch_size=200

    D = Pulse_1stclass_data()
    data_loader = DataLoaderFast(D, batch_size=batch_size, num_workers=0)

    template = 'this is a first trimester ultrasound scan of '
    # template = 'this is a first trimester ultrasound scan of '
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

        assert text_features.shape[0] == class_num * 2 
        logits = (logit_scale * image_features @ text_features[:class_num].t()).detach().softmax(dim=-1)
        logits += (logit_scale * image_features @ text_features[class_num:].t()).detach().softmax(dim=-1)
        predict = torch.argmax(logits, dim=-1).cpu().numpy()

        cm += confusion_matrix(labels.cpu().numpy(), predict, labels = np.array(label_set_num))

        acc_perclass = recall_score(labels.cpu().numpy(), predict, labels = np.array(label_set_num), zero_division=0, average=None)
        l = np.array(labels.cpu().numpy())
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
    plt.savefig('./vis/CM_vis_5classes_norm.png')
    plt.cla()
    plt.clf()
    plt.close('all')   
    # gc.collect()
    xiaoqing

# @torch.no_grad()
# def test_1stclass_pulse(model, device, args):
#     print('Perform pulsedataset 1st trimester classification (contain 83547 images and 6 classes)')
#     model.eval()
#     batch_size=200

#     D = Pulse_1stclass_data()
#     data_loader = DataLoaderFast(D, batch_size=batch_size, num_workers=0)

#     template = 'this is an ultrasound scan of '
#     token = model.tokenizer([template + l for l in label_set], context_length=50).to(device)
#     text_features, logit_scale = model.lang_model(token)

#     Acc_score = 0
#     test_img_len = 0
#     for input_data in tqdm(data_loader, total=len(data_loader)):
#         image_features  = input_data['feat'].to(device)
#         labels  = input_data['label']
#         names  = input_data['name']

#         logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
#         predict = torch.argmax(logits, dim=-1).cpu().numpy()
#         acc = np.sum(predict == np.array(labels)) #/ predict.shape[0] 
#         Acc_score += acc
#         test_img_len += predict.shape[0] 
#         del image_features
#     accuracy = Acc_score/test_img_len
#     print('Classification accuracy: ', accuracy)
#     return accuracy

########### BiomedCLIP classification accuracy (6 classes) #######################
########### BiomedCLIP classification accuracy: 0.3327707757 (segittal)
########### BiomedCLIP classification accuracy: 0.26343255891893186 (sagittal)

## Classification accuracy:  0.26343255891893186 (name more complex)
## Perclass accuracy:  [0.89541241 0.00143021 0.00392465 0.0425815  0.10751819 0.82206497]

## Classification accuracy:  0.21210815469137131 (with first trimester in prompt)
## Perclass accuracy:  [0.88115046 0.         0.         0.08499667 0.00582468 0.85056288]

## Classification accuracy:  0.25092462925060144 (with first trimester in prompt with anatomy graph) 
## Perclass accuracy:  [9.82410269e-01 2.86041190e-04 7.84929356e-04 6.37059215e-02 8.33903364e-02 7.99035060e-01]

########### BiomedCLIP classification accuracy (6 classes) #######################
## BiomedCLIP: 0.4158373336455528 Perclass accuracy:  [0.95578797 0.00243135 0.00902669 0.53725882 0.57366985]
## BiomedCLIP w/ anatomy graph: 0.433594817156461 Perclass accuracy:  [0.99001664 0.00171625 0.00274725 0.27860945 0.89546385]
