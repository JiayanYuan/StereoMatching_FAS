import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from models.yjy_net_original import YJY_Net
from PIL import Image
from models.classification import ClassificationNet
from pyheatmap.heatmap import HeatMap
from dataloader import readpfm as rp
import matplotlib.pyplot as plt
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='weights_yjynet_regression_withoutgroups/checkpoint_199.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= '2020-04-20-16-35-51_IR10_-4l.jpg',
                    help='load model')
parser.add_argument('--rightimg', default= '2020-04-20-16-35-51_IR10_-4r.jpg',
                    help='load model')                                      
parser.add_argument('--model', default='yjy_net_regression',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=24,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
elif args.model == 'yjy_net' or 'yjy_net_regression':
    model = YJY_Net(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    # model.load_state_dict({k.replace('module.',''):v for k,v in state_dict['state_dict'].items()})
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
Net=ClassificationNet()
Net=nn.DataParallel(Net)
Net.cuda()
Net.load_state_dict(torch.load('weights_yjynet_regression_withoutgroups_5w/classification.pth'))
# args.cuda=False
def test(imgL,imgR):
        model.eval()
        Net.eval()
        if args.cuda:
           imgL = imgL.to('cuda')
           imgR = imgR.to('cuda')

        if args.model!='yjy_net' or 'yjy_net_regression':
            with torch.no_grad():
                a=time.time()
                disp = model(imgL,imgR)
                output=Net(disp.unsqueeze(0))
                print('Output',output.softmax(dim=1))
                b=time.time()
                print(b-a)
            disp = torch.squeeze(disp)
            pred_disp = disp.data.cpu().numpy()
        else:
            with torch.no_grad():
                a=time.time()
                disp = model(imgL,imgR)
                b=time.time()
                print(b-a)
            if args.model=='yjy_net':
                pred_disp=disp.permute(0,2,3,1).argmax(-1).squeeze(0)
            if args.model=='yjy_net_regression':
                pred_disp=torch.squeeze(disp)
            pred_disp=pred_disp.data.cpu().numpy()

        return pred_disp


def main():

        # normal_mean_var = {'mean': [0.485, 0.456, 0.406],
        #                     'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              ])
        #
        # imgL_o = Image.open(args.leftimg).convert('RGB')
        # imgR_o = Image.open(args.rightimg).convert('RGB')
        #
        # imgL = infer_transform(imgL_o)
        # imgR = infer_transform(imgR_o)
        imgL=cv2.imread(args.leftimg,cv2.IMREAD_UNCHANGED)
        imgR=cv2.imread(args.rightimg,cv2.IMREAD_UNCHANGED)
        imgL=infer_transform(imgL)
        imgR=infer_transform(imgR)
        imgL=imgL.unsqueeze(0)
        imgR=imgR.unsqueeze(0)
        # pad to width and hight to 16 times
        # if imgL.shape[1] % 16 != 0:
        #     times = imgL.shape[1]//16
        #     top_pad = (times+1)*16 -imgL.shape[1]
        # else:
        #     top_pad = 0
        #
        # if imgL.shape[2] % 16 != 0:
        #     times = imgL.shape[2]//16
        #     right_pad = (times+1)*16-imgL.shape[2]
        # else:
        #     right_pad = 0
        #
        # imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        # imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))
        img=pred_disp
        
        # if top_pad !=0 and right_pad != 0:
        #     img = pred_disp[top_pad:,:-(right_pad+1)]
        # elif right_pad == 0:
        #     img = pred_disp[top_pad:, :]
        # else:
        #     img = pred_disp
        # disp_target = rp.readPFM('0432.pfm')

        plt.matshow(img,cmap=plt.cm.gist_rainbow,vmax=8,vmin=2)
        plt.savefig('prediction_heat.png')

        img = (img*10).astype('uint8')
        # label=cv2.imread('fake_dis.jpg',cv2.IMREAD_UNCHANGED)
        # label_img=(label*10).astype('uint8')
        # cv2.imwrite('fake_label.png',label_img)
        cv2.imwrite('prediction.png',img)

        # plt.matshow(img,cmap=plt.cm.rainbow,vmax=240,vmin=100)
        # plt.savefig('prediction_heat.png')

        # img=img.astype('uint8')
        # img = Image.fromarray(img)
        # img.save('Test_disparity_16.png')

if __name__ == '__main__':
   main()
