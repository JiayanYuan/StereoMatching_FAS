import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models.submodule import *
import time
# from models.refinement import *



class YJY_Net(nn.Module):
    def __init__(self, maxdisp,multiscale=True):
        super(YJY_Net, self).__init__()
        self.maxdisp = maxdisp
        self.multiscale=multiscale

        self.feature=nn.Sequential(
            nn.Conv2d(48, 48, 5, 1, 2, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.Conv2d(48, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.Conv2d(24, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 48, 3, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.PReLU()
        )

        self.SPP1_branch1=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1, groups=1,
                               bias=False)
        )

        self.SPP1_branch2=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False)
        )

        self.SPP1_branch3=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False)
        )

        self.SPP1_branch4=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU()
        )

        self.aggregation=nn.Sequential(nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, padding=1, stride=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.PReLU(),
                                       nn.ConvTranspose2d(in_channels=48, out_channels=48, groups=1, kernel_size=4,
                                                          padding=1, stride=2, bias=False),
                                       nn.PReLU(),
                                       nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, padding=1, stride=1,
                                                 bias=False),
                                       nn.BatchNorm2d(24),
                                       nn.PReLU(),
                                       nn.ConvTranspose2d(in_channels=24, out_channels=24, groups=1, kernel_size=4,
                                                          padding=1, stride=2,bias=False),
                                       nn.PReLU()
                                       )

        # self.Deconv8=nn.ConvTranspose2d(in_channels=48,out_channels=48,groups=1,kernel_size=4,padding=1,stride=2,bias=False)
        # self.prelu8=nn.PReLU()

        # self.Conv9 = nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, padding=1, stride=1, bias=False)
        # self.prelu9 = nn.PReLU()

        # self.Deconv10 = nn.ConvTranspose2d(in_channels=24, out_channels=24, groups=1,kernel_size=4, padding=1, stride=2,
        #                                   bias=False)
        # self.prelu10 = nn.PReLU()



    def forward(self, left, right):


        # return cost_volume
        # left=left_feature[0]
        # right=right_feature[0]

        # cost = left.new_zeros(left.size()[0], int(self.maxdisp),
        #                                  left.size()[2], left.size()[3])
        # for i in range(int(self.maxdisp)):
        #     if i>0:
        #         cost[:,i,:,i:]=(left[:,:,:,i:]*right[:,:,:,:-i]).mean(dim=1)
        #     else:
        #         cost[:, i, :, i:]=(left*right).mean(dim=1)
        #
        cost=Variable(torch.FloatTensor(left.size()[0],left.size()[1]*2,int(self.maxdisp),left.size()[2],left.size()[3]).zero_()).to('cuda')
        # cost = Variable(torch.FloatTensor(left.size()[0], left.size()[1] * 2, int(self.maxdisp), left.size()[2],
        #                                   left.size()[3]).zero_())
        for i in range(int(self.maxdisp)):
            if i > 0:
                cost[:, :left.size()[1], i, :, i:] = left[:, :, :, i:]
                cost[:, left.size()[1]:, i, :, i:] = right[:, :, :, :-i]
            else:
                cost[:, :left.size()[1], i, :, :] = left
                cost[:, left.size()[1]:, i, :, :] = right
        cost = cost.contiguous().view(left.size()[0], left.size()[1] * 2 * int(self.maxdisp),
                                      left.size()[2], left.size()[3])

        x=self.feature(cost)
        branch1=self.SPP1_branch1(x)
        branch2=self.SPP1_branch2(x)
        branch3=self.SPP1_branch3(x)
        branch4=self.SPP1_branch4(x)
        x=torch.cat([branch1,branch2,branch3,branch4],dim=1)
        x=self.aggregation(x)
        x=F.softmax(x,dim=1)
        dis=disparityregression(self.maxdisp)(x)

        return dis

        # if self.multiscale==False:
        #     return dis_branch0
        # else:
        #     dis_branch2=F.softmax(cost_volume[1],dim=1)
        #     max_disp2=dis_branch2.size(1)
        #     disp_candidates2=torch.arange(0,max_disp2).type_as(dis_branch2)
        #     disp_candidates2=disp_candidates2.view(1,max_disp2,1,1)
        #     dis_branch2=torch.sum(dis_branch2*disp_candidates2,1,keepdim=True)
        #     dis_branch2=F.interpolate(dis_branch2,size=(left_image.size(-2),left_image.size(-1)),mode='bilinear',align_corners=False)*(dis_branch2.size(-1)/left_image.size(-1))
        #     dis_branch2=dis_branch2.squeeze(1)
        #
        #     dis_branch3 = F.softmax(cost_volume[2], dim=1)
        #     max_disp3 = dis_branch3.size(1)
        #     disp_candidates3 = torch.arange(0, max_disp3).type_as(dis_branch3)
        #     disp_candidates3 = disp_candidates3.view(1, max_disp3, 1, 1)
        #     dis_branch3 = torch.sum(dis_branch3 * disp_candidates3, 1, keepdim=True)
        #     dis_branch3 = F.interpolate(dis_branch3, size=(left_image.size(-2), left_image.size(-1)), mode='bilinear',
        #                                 align_corners=False) * (dis_branch3.size(-1) / left_image.size(-1))
        #     dis_branch3 = dis_branch3.squeeze(1)
        #
        #     return dis_branch0,dis_branch1,dis_branch2,dis_branch3

        #
        # # branch1 = self.SPP2_branch1(x)
        # # branch2 = self.SPP2_branch2(x)
        # # branch3 = self.SPP2_branch3(x)
        # # branch4 = self.SPP2_branch4(x)
        # # x = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        # # x = self.SPP2_Conv(x)
        # #
        # # branch1 = self.SPP3_branch1(x)
        # # branch2 = self.SPP3_branch2(x)
        # # branch3 = self.SPP3_branch3(x)
        # # branch4 = self.SPP3_branch4(x)
        # # x = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        # # x = self.SPP3_Conv(x)
        #
        # # x=self.prelu7(self.Conv7(x))
        # x=self.prelu8(self.Deconv8(x))
        # x=self.prelu9(self.Conv9(x))
        # x=self.prelu10(self.Deconv10(x))
        #
        # x=F.softmax(x,dim=1)
        # x = disparityregression(self.maxdisp)(x)
        # # x=x.permute(0,2,3,1)
        # # x=x.view(-1,self.maxdisp)
        # return x

if __name__ == '__main__':
    #Resnet40基础网络模型大小为8.3M
    net=YJY_Net(24).cuda()
    left=torch.rand(1,1,256,256).cuda()
    right = torch.rand(1, 1, 256, 256).cuda()
    t=0
    for i in range(10):
        t1=time.time()
        out=net(left,right)
        t2=time.time()
        t=t+t2-t1
        print(t2-t1)
    print('avg',t/10)
    # print(t2-t1)
    # torch.save(net.state_dict(),'net.pth')
    print(out.size())