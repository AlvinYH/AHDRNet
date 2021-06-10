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
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'

parser = argparse.ArgumentParser(description='Attention-guided HDR')

# parser.add_argument('--test_whole_Image', default='./test.txt')
# parser.add_argument('--trained_model_dir', default='./trained-model/')
# parser.add_argument('--trained_model_filename', default='ahdr_model.pt')
parser.add_argument('--list_path', default='./test.txt')
parser.add_argument('--source_dir', default='./test/Extra/')
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--model_path', default='./trained-model/ahdr_model.pt')

# parser.add_argument('--use_cuda', default=True)
# parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

args = parser.parse_args()





class Test_Dataset(Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt) # 10 for test
        # print(self.list_txt)
        # self.transform = transforms.Compose([transforms.Resize(1500, 1000)])
    def __getitem__(self, index):
        sample_path = self.list_txt[index][:-1]
        # print(sample_path)
        filenames = os.listdir(sample_path)
        filenames.sort()
        # print(filenames)
        LDRs = []
        for fn in filenames:
            if 'tif' in fn or 'JPG' in fn:
                image = cv2.imread(os.path.join(sample_path, fn))
                # print(image.shape)
                # image = self.transform(image)
                LDRs.append(torch.tensor(image).permute(2, 0, 1) / 255)
            elif 'hdr' in fn:
                # print(os.path.join(sample_path, fn))
                # HDR = cv2.imread(os.path.join(sample_path, fn), flags=cv2.IMREAD_ANYDEPTH)
                # HDR = torch.tensor(HDR).permute(2, 0, 1)
                pass
            else:
                f = open(os.path.join(sample_path, fn))
                EVpre = f.readlines()
                print(EVpre)
                EV = []
                for i in EVpre:
                    EV.append(float(i[:-1]))
                # print(EV)

        # LDRHDR = torch.cat((LDRs[0], LDRs[1], LDRs[2], HDR), dim=0)
        # LDRHDR = self.transform(LDRHDR)
        return LDRs[0], LDRs[1], LDRs[2], EV, index

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

test_ds = Test_Dataset(args.list_path)
loss_func = nn.L1Loss(reduction='mean')
model = AHDR(args)
# model.apply(weights_init_kaiming)

model = nn.DataParallel(model).cuda()
model.module.load_state_dict(torch.load(args.model_path))


# test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

# opt = optim.Adam(model.parameters(), lr=1e-5)


def tonemap(H, mu):
    return torch.log(1 + mu * H) / torch.log(1 + H)

def preprocess(x, ev):

    return x ** 2.2 / (2 ** ev)

def test(model, dataset, san_check=False):
    len_data = len(dataset)

    elapsed_times = []
    with torch.no_grad():
        for i in range(len_data):
            start = time.time()
            l1, l2, l3, EV, idx = dataset[i]
            # Todo:这里的EV直接设为0，2，4
            EV = [0, 2, 4]
            l1 = l1.unsqueeze(0)
            l2 = l2.unsqueeze(0)
            l3 = l3.unsqueeze(0)
            print(torch.min(l1), torch.max(l1), torch.min(l2), torch.max(l2), torch.min(l3), torch.max(l3))
            l1 = torch.cat((l1, preprocess(l1, EV[0])), dim=1).cuda()
            l2 = torch.cat((l2, preprocess(l2, EV[1])), dim=1).cuda()
            l3 = torch.cat((l3, preprocess(l3, EV[2])), dim=1).cuda()
            
            # print(torch.min(l1), torch.max(l1), torch.min(l2), torch.max(l2), torch.min(l3), torch.max(l3))

            pred = model(l1, l2, l3)[0]
            
            pred = pred.permute(1, 2, 0).cpu()
            # torch.save(pred, "pttestHDR" + str(idx) + ".pt")
            cv2.imwrite(args.result_dir + "testHDR" + str(idx) + ".hdr", pred.numpy())

            elapsed = time.time() - start
            elapsed_times.append(elapsed)
            if san_check is True:
                break
    inference_time = np.mean(elapsed_times) * 1000
    print("average inference time per image: %.2f ms "
          % (inference_time))
    return



test(model, test_ds, san_check=False)


