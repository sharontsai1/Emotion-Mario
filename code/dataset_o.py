import torch
import torch.utils.data as data
import imageio
import os
import sys
import json
import numpy as np

def np2PytorchTensor(img, label):
    ts = (2, 0, 1)
    img = torch.Tensor(img.transpose(ts).astype(float))
    label = torch.Tensor(np.array(label).astype(float)).type(torch.LongTensor)

    return img, label

class dataset(data.Dataset):
    def __init__(self, batchsize, eventDir, imgDir):
        self.batchsize = batchsize
        self.eventDir = eventDir
        self.imgDir = imgDir
        with open(self.eventDir) as json_file:
            data = json.load(json_file)
        self.eventList = data
    
    def __getitem__(self, idx):
        player = self.eventList[idx]['player']
        framename = self.eventList[idx]['frame_number']
        label = self.eventList[idx]['event']

        img_path = '{}/{}/{}_game/game_{}.png'.format(self.imgDir, player, player, framename)
        img = imageio.imread(img_path) / 255.0

        return np2PytorchTensor(img, label)
    
    def __len__(self):
        return len(self.eventList)