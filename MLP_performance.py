#!/usr/bin/env python
import setGPU
import torch.nn as nn
import torch
import jet_dataset
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import json
import yaml
import numpy as np
from datetime import datetime
import os
import os.path as path
from optparse import OptionParser
import pandas as pd
import hls4ml
from sklearn.metrics import accuracy_score, roc_curve, auc
import sys

def roc_data(y, predict_test, labels):
    df = pd.DataFrame()
    fpr = {}
    tpr = {}
    auc1 = {}

    for i, label in enumerate(labels):
        df[label] = y[:,i]
        df[label + '_pred'] = predict_test[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])

        auc1[label] = auc(fpr[label], tpr[label])
    return fpr, tpr, auc1

## Load yaml config
def parse_config(config_file) :

    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config, Loader=yaml.FullLoader)

yamlConfig = parse_config("yamlConfig.yml")
labels = yamlConfig['Labels']

# Setup test data set
test_dataset = jet_dataset.ParticleJetDataset("./train_data/test/", yamlConfig)
test_size = len(test_dataset)

print("test dataset size: " + str(len(test_dataset)))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size,
                                          shuffle=False, num_workers=10, pin_memory=True)

X_test = None
y_test = None

for i, data in enumerate(test_loader, 0):
    X_test, y_test = data[0].numpy(), data[1].numpy()

class three_layer_model_batnorm_masked(nn.Module):
    def __init__(self, bn_affine = True, bn_stats = True ):
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model_batnorm_masked, self).__init__()
        self.quantized_model = False
        self.input_shape = 16  # (16,)
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.bn1 = nn.BatchNorm1d(64, affine=bn_affine, track_running_stats=bn_stats)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32, affine=bn_affine, track_running_stats=bn_stats)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32, affine=bn_affine, track_running_stats=bn_stats)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 5)
        self.softmax = nn.Softmax(0)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.softmax(x)

        return x

dict_model = torch.load("32b_70Pruned_0rand.pth", map_location=lambda storage, loc: storage)
model = three_layer_model_batnorm_masked()
model.load_state_dict(dict_model)

x = torch.from_numpy(X_test)
x = x.to(torch.float32)
y_pred_pt = model(x).detach().cpu().numpy()

config = hls4ml.utils.config_from_pytorch_model(model, granularity='model')

config['Model']['Precision'] = 'ap_fixed<16,6>'

hls_model = hls4ml.converters.convert_from_pytorch_model(model, input_shape = [None,16], hls_config=config)

hls_model.compile()

y_pred = hls_model.predict(X_test)
print("hls4ml {} Accuracy: {}".format(config['Model']['Precision'],accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))))
print("PyTorch Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_pt, axis=1))))

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

fpr_hls, tpr_hls, auc_hls = roc_data(y_test, y_pred, labels=labels)
for l in labels:
    idx, val = find_nearest(tpr_hls[l], 0.5)
    print('hls4ml', l, 'eff_bkg @ eff_sig=0.5:', fpr_hls[l][idx])
    print('hls4ml', l, 'auc:                  ', auc_hls[l])
fpr_hls_ave = np.average([fpr_hls[l][idx] for l in labels])
auc_hls_ave = np.average([auc_hls[l] for l in labels])
print('hls4ml average', 'eff_bkg @ eff_sig=0.5:', fpr_hls_ave)
print('hls4ml average', 'auc                  :', auc_hls_ave)

fpr_pt, tpr_pt, auc_pt = roc_data(y_test, y_pred_pt, labels=labels)
for l in labels:
    idx, val = find_nearest(tpr_pt[l], 0.5)
    print('PyTorch', l, 'eff_bkg @ eff_sig=0.5:', fpr_pt[l][idx])
    print('PyTorch', l, 'auc:                  ', auc_pt[l])
fpr_pt_ave = np.average([fpr_pt[l][idx] for l in labels])
auc_pt_ave = np.average([auc_pt[l] for l in labels])
print('PyTorch average', 'eff_bkg @ eff_sig=0.5:', fpr_pt_ave)
print('PyTorch average', 'auc                  :', auc_pt_ave)

####-------------ONNX TEST---------------####
sys.exit()
import onnx
onnx_model = onnx.load('32b_70Pruned_FullModel.onnx')

onnx_config = hls4ml.utils.config_from_onnx_model(onnx_model, granularity='model')
hls_model_onnx = hls4ml.converters.convert_from_onnx_model(onnx_model, hls_config=onnx_config)

hls_model_onnx.compile()

y_pred_onnx = hls_model_onnx.predict(X_test)
print("ONNX Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_onnx, axis=1))))
fpr_hls, tpr_hls, auc_hls = roc_data(y_test, y_pred_onnx, labels=labels)
for l in labels:
    idx, val = find_nearest(tpr_hls[l], 0.5)
    print('ONNX', l, 'eff_bkg @ eff_sig=0.5:', fpr_hls[l][idx])
    print('ONNX', l, 'auc:                  ', auc_hls[l])
fpr_hls_ave = np.average([fpr_hls[l][idx] for l in labels])
auc_hls_ave = np.average([auc_hls[l] for l in labels])
print('ONNX average', 'eff_bkg @ eff_sig=0.5:', fpr_hls_ave)
print('ONNX average', 'auc                  :', auc_hls_ave)
