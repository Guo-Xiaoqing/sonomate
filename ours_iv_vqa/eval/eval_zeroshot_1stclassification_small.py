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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content

name_corr = {
    'AbdoCI':'abdomen', 
    'Cardiac':'cardiac', 
    'Head':'head', 
    'LowerLimb':'lower limb', 
    'Spine':'spine', 
    'UpperLimb':'upper limb'
    }
label_set = ['abdomen', 'cardiac', 'head', 'lower limb', 'spine', 'upper limb']
name_corr_num = {'abdomen':0, 'cardiac':1, 'head':2, 'lower limb':3, 'spine':4, 'upper limb':5}
class_dist = {'abdomen':100, 'cardiac':100, 'head':100, 'lower limb':100, 'spine':100, 'upper limb':99}

class_num = 6
label_set_abb = ['ABD', 'Cardiac', 'BPD', 'LowerLimb', 'CRL', 'UpperLimb']
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

feature_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/small_1st_class/First_traimster_Small_Classfication_Dataset/'
label_path = '/media/engs2527/2TBFast/sonomate/vqa_dataset/small_1st_class/First_traimster_Small_Classfication_Dataset.txt'

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
def test_1stclass_pulse_small(model, device, args):
    print('Perform pulsedataset 1st trimester small classification (contain 599 images and 6 classes)')
    model.eval()
    batch_size=200

    D = Pulse_1stclass_data()
    data_loader = DataLoaderFast(D, batch_size=batch_size, num_workers=0)

    # template = 'this is a first trimester ultrasound scan of '
    template = 'this is an ultrasound scan of '
    # token = model.tokenizer([template + l for l in label_set], context_length=50).to(device)

    class_template = [template + l for l in label_set]
    class_template_complex = [template + l for l in label_set_complex]
    class_template.extend(class_template_complex)

    token = model.tokenizer(class_template, context_length=32).to(device)
    text_features, logit_scale = model.lang_model(token)

    Acc_score = 0
    test_img_len = 0
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

        del image_features
    accuracy = Acc_score/test_img_len
    accuracy_perclass = Acc_perclass_score/test_perclass_len
    print('Classification accuracy: ', accuracy, 'Perclass accuracy: ', accuracy_perclass)
    return accuracy#, accuracy_perclass




## Classification accuracy:  0.21535893155258765 (with first trimester in prompt)
## Perclass accuracy:  [0.         0.14       0.92       0.         0.01       0.22222222]

## Classification accuracy:  0.2587646076794658 (with first trimester in prompt with anatomy graph) 
## Perclass accuracy:  [0.        0.23      0.74      0.4       0.08      0.1010101]

## 