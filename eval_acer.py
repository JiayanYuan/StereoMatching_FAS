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
parser.add_argument('--model', default='basic',
                    help='select model')
parser.add_argument('--datapath', default='/data1/SceneFlow/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='weights/Net.pth',
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

# Net=YJY_Net(args.maxdisp,multiscale=False)
Net=basic(args.maxdisp)
Net=torch.nn.DataParallel(Net)
Net=Net.to('cuda')
Net.load_state_dict(torch.load('weights/Net.pth')['state_dict'])
device='cuda'
model=ClassificationNet()
model=torch.nn.DataParallel(model)
model=model.to(device)
model.load_state_dict(torch.load('weights/classification.pth'))
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
if not os.path.exists(args.savemodel):
    os.mkdir(args.savemodel)


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
print(model)

def eval_acer(results,live,spoof):
    """
    :param results: np.array shape of (N, 2) [pred, label]
    :param is_print: print eval score
    :return: score
    """
    ind_n = (results[:, 1] == spoof)
    ind_p = (results[:, 1] == live)
    fp = (results[ind_n, 0] == live).sum()
    fn = (results[ind_p, 0] == spoof).sum()
    apcer = fp / ind_n.sum() * 100
    bpcer = fn / ind_p.sum() * 100
    acer = (apcer + bpcer) / 2

    print('***************************************')
    print('APCER    BPCER     ACER')
    print('{:.4f}   {:.4f}   {:.4f}'.format(apcer, bpcer, acer))
    print('***************************************')




def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    results_gt=torch.Tensor()
    results_pred=torch.Tensor()
    with torch.no_grad():
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, targets) in enumerate(TestImgLoader):
            disp = inference(imgL_crop, imgR_crop)
            targets=targets.to(device)
            targets = targets.long()
            outputs = model(disp)
            _, predicted = outputs.max(1)
            results_gt=torch.cat([results_gt.cpu().int(),targets.cpu().int()])
            results_pred=torch.cat([results_pred.cpu().int(),predicted.cpu().int()])
    results=torch.cat([results_pred.unsqueeze(1),results_gt.unsqueeze(1)],dim=1)
    results=results.int().numpy()

    eval_acer(results,0,1)

def inference(imgL,imgR):
    Net.eval()
    imgR=imgR.to('cuda')
    imgL=imgL.to('cuda')
    disp = Net(imgL, imgR)
    disp=disp.unsqueeze(1)
    return disp
def main():
    test()
if __name__ == '__main__':
    main()