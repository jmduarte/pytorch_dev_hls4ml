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

import hls4ml


## Load yaml config
def parse_config(config_file) :

    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config, Loader=yaml.FullLoader)

yamlConfig = parse_config("yamlConfig.yml")

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
print(x)
print(y_pred_pt)

config = hls4ml.utils.config_from_pytorch_model(model, granularity='model')

config['Model']['Precision'] = 'ap_fixed<16,6>'

hls_model = hls4ml.converters.convert_from_pytorch_model(model, input_shape = [None,16], hls_config=config)

hls_model.compile()

from sklearn.metrics import accuracy_score
y_pred = hls_model.predict(X_test)
print("hls4ml {} Accuracy: {}".format(config['Model']['Precision'],accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))))
print("PyTorch Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_pt, axis=1))))

####-------------ONNX TEST---------------####
import onnx
onnx_model = onnx.load('32b_70Pruned_FullModel.onnx')

onnx_config = hls4ml.utils.config_from_onnx_model(onnx_model, granularity='model')
hls_model_onnx = hls4ml.converters.convert_from_onnx_model(onnx_model, hls_config=onnx_config)

hls_model_onnx.compile()

y_pred_onnx = hls_model_onnx.predict(X_test)
print("ONNX Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_onnx, axis=1))))
