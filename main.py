import os
import sys

from dataset import get_data
from get_model import get_configured_model
from models.Informer import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from models.pyraformer import Pyraformer_LR
# from models.reformer_pytorch.reformer_pytorch import Reformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy import stats
warnings.filterwarnings('ignore')
import pdb
import pickle

import os


model, args = get_configured_model()

train_data, train_loader =  get_data()


time_now = time.time()

train_steps = len(train_loader)
#early_stopping = EarlyStopping(patience=.args.patience, verbose=True)

learning_rate =0.001
train_epochs = 10
model_optim = optim.Adam(model.parameters(), lr= learning_rate)
criterion = nn.MSELoss()


for epoch in range(train_epochs):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        iter_count += 1
        model_optim.zero_grad()
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # if self.args.add_noise_train:
        #     batch_x = batch_x + 0.3 * torch.from_numpy(np.random.normal(0, 1, size=batch_x.float().shape)).float().to(
        #         self.device)

        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()

        # encoder - decoder
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

        f_dim = -1 #if self.args.features == 'MS' else 0
        # print(batch_y.shape)
        batch_y = batch_y[:, -args.pred_len:, f_dim:]
        # print(batch_y.shape)
        # print(outputs.shape)
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((train_epochs - epoch) * train_steps - i)
            # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        loss = loss.clone()
        loss.backward()
        model_optim.step()

    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_loss = np.average(train_loss)
    # vali_loss = self.vali(vali_data, vali_loader, criterion)
    # test_loss = self.vali(test_data, test_loader, criterion)
    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  ".format(
            epoch + 1, train_steps, train_loss ))
    # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
    #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    # early_stopping(vali_loss, self.model, path)
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break

    #adjust_learning_rate(model_optim, epoch + 1, self.args)

best_model_path = 'best_model.pth' #path + '/' + 'checkpoint.pth'
model.load_state_dict(torch.load(best_model_path))