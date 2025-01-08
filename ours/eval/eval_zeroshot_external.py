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

label_set = ['Fetal abdomen', 'Fetal brain', 'Maternal cervix', 'Fetal femur', 'Fetal thorax', 'Other']
name_corr_num = {'Fetal abdomen':0, 'Fetal brain':1, 'Maternal cervix':2, 'Fetal femur':3, 'Fetal thorax':4, 'Other':5}
class_dist = {'Fetal abdomen':711, 'Fetal brain':3092, 'Maternal cervix':1626, 'Fetal femur':1040, 'Fetal thorax':1718, 'Other':4213}

class_num = 6
label_set_abb = ['Fetal abdomen', 'Fetal brain', 'Maternal cervix', 'Fetal femur', 'Fetal thorax', 'Other']
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
# name_corr_keys = list(name_corr.keys())

# os.environ['CUDA_VISIBLE_DEVICES']='0'

feature_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_external_dataset/test_external_dataset/'
label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/test_external_dataset/test_external_dataset_test.txt'

class Pulse_2ndclass_data():
    """Ultrasound external classification dataset. 
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
def test_external(model, device, args):
    # print('Perform ultrasound external classification (contain 12400 images and 6 classes)')
    # print('Perform ultrasound external classification (contain 7129 images and 6 classes)')
    print('Perform ultrasound external classification (contain 5244 images and 6 classes)')
    model.eval()
    batch_size=200

    D = Pulse_2ndclass_data()
    data_loader = DataLoaderFast(D, batch_size=batch_size, num_workers=0)

    template = 'an ultrasound scan containing '
    # template = 'this is an ultrasound scan of '
    # token = model.tokenizer([template + l for l in label_set], context_length=50).to(device)

    class_template = [template + l for l in label_set]
    class_template_complex = [template + l for l in label_set_complex]
    class_template.extend(class_template_complex)

    # token = model.tokenizer(class_template, context_length=args.context_length*2).to(device)
    token = model.tokenizer(class_template, context_length=77).to(device)
    text_features, logit_scale = model.lang_model(token)
    text_features[class_num-1,:] = text_features[-1,:]

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
    plt.savefig('./vis/CM_vis_external_norm.png')
    plt.cla()
    plt.clf()
    plt.close('all')   
    # gc.collect()
    # xiaoqing

########### BiomedCLIP test classification accuracy: 0.23502556610664718

## Whole dataset BiomedCLIP: 0.6237903225806452 Perclass accuracy:  [0.05766526 0.99353169 0.99138991 0.37211538 0.81781141 0.28910515]
## Whole dataset BiomedCLIP w/ test graph: 0.6795161290322581 Perclass accuracy:  [0.00843882 0.94372574 0.98708487 0.30865385 0.80500582 0.52053169]

## Training dataset BiomedCLIP: 0.6411838967597139 Perclass accuracy:  [0.01133144 0.90185185 0.98674822 0.21705426 0.88374291 0.41945406]
## Training dataset BiomedCLIP w/ test graph: 0.6724645812876981 Perclass accuracy:  [0.00566572 0.93950617 0.98470948 0.33333333 0.78638941 0.49980777]

## Test dataset BiomedCLIP: 0.5966609751470309 Perclass accuracy:  [0.01675978 0.92595109 0.99224806 0.24618321 0.87727273 0.26550868]
## Test dataset BiomedCLIP w/ test graph: 0.6890533105672548 Perclass accuracy:  [0.01117318 0.94836957 0.99069767 0.28435115 0.83484848 0.55397022]
