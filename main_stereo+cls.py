from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import KITTIloader2012 as kitti12
from dataloader import SecenFlowLoader as DA
# from dataloader import KITTILoader as DA
from dataloader import FaceLoader_classification as FL
from models.classification import ClassificationNet
from models import *
from models.yjy_net_original import YJY_Net
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=24,
                    help='maxium disparity')
parser.add_argument('--model', default='yjy_net',
                    help='select model')
parser.add_argument('--datapath', default='/data1/SceneFlow/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=60,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='weights_yjynet_regression_withoutgroups_5w/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
start_epoch=0
# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = kitti12.dataloader(args.datapath)
# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)
all_left_img=[]
all_right_img=[]
all_left_disp=[]
all_left_classification=[]

test_left_img=[]
test_right_img=[]
test_left_disp=[]
test_left_classification=[]
for line in open('real_train_5w.txt','r'):
    left=line.split(' ')[0]
    right=line.split(' ')[1]
    dis=line.split(' ')[2].replace('\n','')
    all_left_img.append(left)
    all_right_img.append(right)
    all_left_disp.append(dis)
    all_left_classification.append(0)
for line in open('fake_train_5w.txt','r'):
    left=line.split(' ')[0]
    right=line.split(' ')[1]
    dis=line.split(' ')[2].replace('\n','')
    all_left_img.append(left)
    all_right_img.append(right)
    all_left_disp.append(dis)
    all_left_classification.append(1)
for line in open('real_test_5w.txt','r'):
    left=line.split(' ')[0]
    right=line.split(' ')[1]
    dis=line.split(' ')[2].replace('\n','')
    test_left_img.append(left)
    test_right_img.append(right)
    test_left_disp.append(dis)
    test_left_classification.append(0)
for line in open('fake_test_5w.txt','r'):
    left=line.split(' ')[0]
    right=line.split(' ')[1]
    dis=line.split(' ')[2].replace('\n','')
    test_left_img.append(left)
    test_right_img.append(right)
    test_left_disp.append(dis)
    test_left_classification.append(1)


TrainImgLoader = torch.utils.data.DataLoader(
         FL.myImageFloder(all_left_img,all_right_img,all_left_disp,all_left_classification, True),
         batch_size= 128, shuffle= True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         FL.myImageFloder(test_left_img,test_right_img,test_left_disp,test_left_classification, False),
         batch_size= 128, shuffle= False, num_workers= 8, drop_last=False)
# TestImgLoader = torch.utils.data.DataLoader(
#          DA.myImageFloder(all_left_img,all_right_img,all_left_disp, False),
#          batch_size= 1, shuffle= False, num_workers= 2, drop_last=False)
print('Number of train Image: ',len(TrainImgLoader))
Net=YJY_Net(args.maxdisp,multiscale=False)
Net=torch.nn.DataParallel(Net)
Net=Net.to('cuda')
Net.load_state_dict(torch.load('weights_yjynet_regression_withoutgroups/checkpoint_199.tar')['state_dict'])
device='cuda'
model=ClassificationNet()
model=torch.nn.DataParallel(model)
model=model.to(device)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
if not os.path.exists(args.savemodel):
    os.mkdir(args.savemodel)


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer1=optim.Adam(Net.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer2 = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
print(model)

def train_stereo(imgL,imgR,disp_L):
    Net.train()
    optimizer1.zero_grad()
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    mask = disp_true < args.maxdisp
    mask.detach_()
    output = Net(imgL, imgR)
    # output = torch.squeeze(output, 1)
    loss = F.smooth_l1_loss(output[mask], disp_true[mask] * 1.0, size_average=True)
    loss.backward()
    optimizer1.step()
    return loss.data


def test_stereo(imgL,imgR,disp_L):
    Net.eval()
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    mask = disp_true < args.maxdisp
    output = Net(imgL, imgR)
    loss = F.l1_loss(output[mask], disp_true[mask] * 1.0)
    return loss.data.cpu()

def train(epoch):
    # print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, targets) in enumerate(TrainImgLoader):
        disp = inference(imgL_crop, imgR_crop)
        targets=targets.to(device)
        targets=targets.long()
        optimizer2.zero_grad()
        outputs = model(disp)
        # print(net)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer2.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.*correct/total

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, targets) in enumerate(TestImgLoader):
            disp = inference(imgL_crop, imgR_crop)
            targets=targets.to(device)
            targets = targets.long()
            outputs = model(disp)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.*correct/total

def inference(imgL,imgR):
    Net.eval()
    imgR=imgR.to('cuda')
    imgL=imgL.to('cuda')
    disp = Net(imgL, imgR)
    disp=disp.unsqueeze(1)
    return disp


def adjust_learning_rate(optimizer, epoch):

    if epoch<=25:
        lr=0.001
    elif epoch<=40:
        lr=0.0005
    else:
        lr=0.0001
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    for epoch in range(start_epoch,args.epochs):
        print('This is stereo %d-th epoch'%(epoch))
        total_train_loss=0
        adjust_learning_rate(optimizer1,epoch)

        for batch_idx,(imgL_crop,imgR_crop,disp_crop_L,targets) in enumerate(TrainImgLoader):
            loss=train_stereo(imgL_crop,imgR_crop,disp_crop_L)
            # print('Iter %d test loss = %.3f' % (batch_idx, loss))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        total_test_loss=0
        for batch_idx, (imgL, imgR, disp_L,targets) in enumerate(TestImgLoader):
            test_loss = test_stereo(imgL, imgR, disp_L)
            # print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
            total_test_loss += test_loss

        print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))

        savefilename = args.savemodel + '/stereo.pth'
        torch.save({
            'epoch': epoch,
            'state_dict': Net.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)
    for epoch in range(start_epoch,args.epochs):
        print('This is classification %d-th epoch'%(epoch))

        adjust_learning_rate(optimizer2,epoch)
        train_acc = train(epoch)
        test_acc = test(epoch)

        print('Epoch:%d' % epoch, 'Train acc: %.2f%%' % train_acc, 'Test acc: %.2f%%' % test_acc)
        torch.save(model.state_dict(), args.savemodel+'classification.pth')


if __name__ == '__main__':
    main()
