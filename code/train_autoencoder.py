import string
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import pytorch_ssim
from math import log10
from model import ResNet50, ResNet101, autoencoder
from dataset_autoencoder import dataset_Autoencoder as dataset
from dataset_autoencoder import dataset_Autoencoder_test
import argparse
import os
import numpy
import random
import pandas as pd

#===== Training settings =====#
parser = argparse.ArgumentParser(description='ResNet50-predict event')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs for training')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use, if Your OS is window, please set to 0')
parser.add_argument('--seed', type=int, default=777, help='random seed to use. Default=777')
parser.add_argument('--printEvery', type=int, default=10, help='number of batches to print average loss ')
parser.add_argument('--eventDir_train', type=str, default='../train_data/event/event.json', help='path of training set event')
parser.add_argument('--imgDir_train', type=str, default='../train_data/data', help='path of training set image')
parser.add_argument('--eventDir_val', type=str, default='../val_data/event/event.json', help='path of validation set event')
parser.add_argument('--imgDir_val', type=str, default='../val_data/data', help='path of validation set image')
parser.add_argument('--test', action='store_true', default=False, help='test of testing data')
parser.add_argument('--resume', action='store_true', default=False, help='resume training')
parser.add_argument('--eventDir_test', type=str, default='../test_data/event/event.json', help='path of testing set event')
parser.add_argument('--imgDir_test', type=str, default='../test_data/data/participant_2', help='path of testing set image')
parser.add_argument('--load_model_path', type=str, default='../model_trained_autoencoder_lr0.0005/epoch_149.pth', help='path of testing set image')
args = parser.parse_args()

print(args)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)
torch.backends.cudnn.benchmark = False

#===== Datasets =====#
def seed_worker(worker_id):
    worker_seed = args.seed
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    
print('===> Loading datasets')
train_set = dataset(batchsize = args.batchSize, eventDir = args.eventDir_train, imgDir = args.imgDir_train)
train_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True, worker_init_fn=seed_worker)
val_set = dataset(batchsize = args.batchSize, eventDir = args.eventDir_val, imgDir = args.imgDir_val)
val_data_loader = DataLoader(dataset=val_set, num_workers=args.threads, batch_size=1, shuffle=False, worker_init_fn=seed_worker)
test_set = dataset_Autoencoder_test(batchsize = args.batchSize, eventDir = args.eventDir_test, imgDir = args.imgDir_test)
test_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False, worker_init_fn=seed_worker)

print('\ttrain data number:', len(train_set))
print('\tvalidaton data number:', len(val_set))
print('\ttesting data number:', len(test_set))

#===== ZebraSRNet model =====#
print('===> Building model')
net = autoencoder()
if args.test or args.resume:
    net = torch.load(args.load_model_path)

if args.cuda:
    net = net.cuda()
print(net)

#===== Loss function and optimizer =====#
#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
#criterion = pytorch_ssim.SSIM()

if args.cuda:
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(net.parameters(), args.lr)

#===== Training and validation procedures =====#
def train(f, epoch):
    net.train()
    epoch_loss = 0
    for iteration, batch in enumerate(train_data_loader):
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        optimizer.zero_grad()
        loss = criterion(net(varIn), varTar)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        
        if (iteration+1)%args.printEvery == 0:
            print("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}".format(epoch, iteration+1, len(train_data_loader), epoch_loss/args.printEvery))
            f.write("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}\n".format(epoch, iteration+1, len(train_data_loader), epoch_loss/args.printEvery))
            epoch_loss = 0

def validate(f):
    net.eval()
    test_loss, correct = 0, 0
    for batch in val_data_loader:
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        prediction = net(varIn)
        # loss
        test_loss += criterion(prediction, varTar).item()

    test_loss /= len(val_data_loader)
    print(f"Val Error: \n\tAvg loss: {test_loss:>8f} \n")
    f.write(f"Val Error: \n\tAvg loss: {test_loss:>8f} \n")

def test(f):
    net.eval()
    anomality = list()
    number = 1
    for batch in test_data_loader:
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        prediction = net(varIn)
        # loss
        test_loss = criterion(prediction, varIn).item()

        anomality.append(test_loss)
        print(number, test_loss)
        number += 1

    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['Predicted'])
    df.to_csv('{}/anomality.csv'.format(args.imgDir_test), index_label = 'Id')

def checkpoint(epoch): 
    save_name = 'epoch_{}.pth'.format(epoch)
    save_path = os.path.join(save_dir, save_name)
    torch.save(net, save_path)
    print("Checkpoint saved to {}".format(save_path))

#===== Main procedure =====#
with open('../log/test_autoencoer.log','w') as f:
    f.write('random seed={}\n'.format(args.seed))
    f.write('dataset configuration: batch size = {}\n'.format(args.batchSize))
    print('-------')
    if args.test:
        test(f)
    else:
        #===== Model saving =====#
        save_dir = '../model_trained_autoencoder_lr{}'.format(args.lr)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for epoch in range(1, args.nEpochs+1):
            train(f, epoch)
            validate(f)
            checkpoint(epoch)
