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

class maxpool1d_model(nn.Module):
    def __init__(self):
        super(maxpool1d_model, self).__init__()
        self.mp1d = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.mp1d(x)
        return x

# Pooling layer 
input_shape = (10, 2) # (Channels, Length)

model = maxpool1d_model()

# Let's use a batch of 1
input_shape_batch = (1,) + input_shape
X_test = torch.from_numpy(np.random.rand(*(input_shape_batch)))
X_test = X_test.to(torch.float32)
y_pred_pt = model(X_test).detach().cpu().numpy()

config = hls4ml.utils.config_from_pytorch_model(model, granularity='model')
config['Model']['Precision'] = 'ap_fixed<16,6>'

hls_model = hls4ml.converters.convert_from_pytorch_model(model, 
                                                         input_shape = [None, input_shape[0], input_shape[1]], 
                                                         hls_config=config)

hls_model.compile()

from sklearn.metrics import accuracy_score
X_test = X_test.detach().cpu().numpy()
y_pred = hls_model.predict(X_test)
y_pred = y_pred.reshape(y_pred_pt.shape)

print('X_test', X_test.shape) 
print(X_test)
print('y_pred_pt', y_pred_pt.shape)
print(y_pred_pt)
print('y_pred', y_pred.shape)
print(y_pred)

df = pd.DataFrame({'PyTorch': y_pred_pt.flatten(), 'hls4ml': y_pred.flatten()})
print(df)
