import pandas as pd
from PIL import Image
import torch
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as transforms
import os
import torchvision.transforms as transforms
import numpy as np
import sys
from torchvision import models
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
import copy
import time
import cv2
import random


from model import *
from running_func import *
from utils import *

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
# Todo: 这里要改成可用的GPU号
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'

parser = argparse.ArgumentParser(description='Attention-guided HDR')

# parser.add_argument('--test_whole_Image', default='./test.txt')
# parser.add_argument('--trained_model_dir', default='./trained-model/')
# parser.add_argument('--trained_model_filename', default='ahdr_model.pt')
# parser.add_argument('--result_dir', default='./result/')
# parser.add_argument('--use_cuda', default=True)
# parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)
parser.add_argument('--list_path', default='./train.txt')
parser.add_argument('--model_name', default='./models/xxx.pt')

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

args = parser.parse_args()





class Train_Dataset(Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt) # 10 for test
        # print(self.list_txt)
        self.transform = transforms.Compose([transforms.RandomCrop(256)])
    def __getitem__(self, index):
        sample_path = self.list_txt[index][:-1]

        filenames = os.listdir(sample_path)
        filenames.sort()
        # print(filenames)
        LDRs = []
        for fn in filenames:
            if 'tif' in fn:
                image = cv2.imread(os.path.join(sample_path, fn))
                # print(image.shape)
                LDRs.append(torch.tensor(image).permute(2, 0, 1) / 255)
            elif 'hdr' in fn:
                # print(os.path.join(sample_path, fn))
                HDR = cv2.imread(os.path.join(sample_path, fn), flags=cv2.IMREAD_ANYDEPTH)
                HDR = torch.tensor(HDR).permute(2, 0, 1)
            else:
                f = open(os.path.join(sample_path, fn))
                EVpre = f.readlines()
                EV = []
                for i in EVpre:
                    EV.append(float(i[:-1]))
                # print(EV)

        LDRHDR = torch.cat((LDRs[0], LDRs[1], LDRs[2], HDR), dim=0)
        LDRHDR = self.transform(LDRHDR)
        return LDRHDR[0:9], LDRHDR[9:], EV

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

train_ds = Train_Dataset(args.list_path)
loss_func = nn.L1Loss(reduction='mean')
model = AHDR(args)
model.apply(weights_init_kaiming)
model = nn.DataParallel(model).cuda()

# print(type(loss_func))

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

opt = optim.Adam(model.parameters(), lr=1e-5)


def tonemap(H, mu):
    return torch.log(1 + mu * H) / torch.log(1 + mu)
def loss_batch(output, loss_func, gt, opt):
    # print(type(loss_func))
    loss = loss_func(tonemap(output, 5000), tonemap(gt, 5000))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

def loss_epoch(model, loss_func, dl, opt, san_check=False):
    running_loss = 0.0
    len_data = len(dl.dataset)
    for LDRs, HDR, EV in dl:
        # print(LDRs.shape, HDR.shape)
        LDR = [LDRs[:, 0:3].cuda(), LDRs[:, 3:6].cuda(), LDRs[:, 6:9].cuda()]
        LDR1 = torch.cat((LDR[0], preprocess(LDR[0], EV[0])), dim=1)
        LDR2 = torch.cat((LDR[1], preprocess(LDR[1], EV[1])), dim=1)
        LDR3 = torch.cat((LDR[2], preprocess(LDR[2], EV[2])), dim=1)
        # print(LDR1.shape, LDR2.shape, LDR3.shape)
        print("LDR", torch.min(LDR1), torch.max(LDR1))
        output = model(LDR1, LDR2, LDR3)
        print(output[0][0])
        print("HDR", torch.min(HDR), torch.max(HDR))
        loss_b = loss_batch(output, loss_func, HDR.cuda(), opt)
        running_loss += loss_b
        if san_check is True:
            break
    loss = running_loss / float(len_data)
    return loss

def preprocess(x, ev):
    for num, i in enumerate(x):
        i = i ** 2.2
        i /= (2 ** ev[num])
    return x

def train(model, params):
    # unpack the parameters
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    sanity_check = params["san_check"]
    path2weights = params["path2weights"]

    # print(type(loss_func))
    # define the log dictionaries
    loss_log = []
    # the variables that record the best performance
    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # training
        t = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        model.train()
        train_loss = loss_epoch(model, loss_func, train_dl, opt, sanity_check)
        loss_log.append(train_loss)

        # validation


        # recording
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")

        # update the lr

        # show
        print("train loss: %.6f"
              % (train_loss))
        print("-" * 10)
        print(time.time() - t)

    # save
    model.load_state_dict(best_model_wts)
    np.save('losslog.npy', loss_log)
    return model, loss_log

os.makedirs("./models", exist_ok=True)
params_train={
"num_epochs": 10,
"optimizer": opt,
"loss_func": loss_func,
"train_dl": train_dl,
"san_check": False,
"path2weights": args.model_name,
}
model, loss_log = train(model, params_train)

