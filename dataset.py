#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:52:44 2018

@author: xiang
"""
import random
import torch
import torchvision
import torch.utils.data as data 
import cv2
import numpy as np
import pickle
from utils import get_imglists, rotatepoints, procrustes, draw_gaussian, enlarge_box, flippoints, get_gtbox, show_image, loadFromPts


class Dataset(data.Dataset): # torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override these methods
    def __init__(self, imgdirs, phase, attr, rotate, res=128, gamma=3, target_type='heatmap'):
        
        self.imglists = get_imglists(imgdirs)
        assert phase in ['train', 'test'], 'Only support train and test'
        self.phase = phase
        self.r = rotate
        self.res = res
        assert target_type in ['heatmap','landmarks'], 'Only support heatmap regression and landmarks regression'
        self.target_type = target_type
        self.gamma = gamma
        self.transform = torchvision.transforms.ToTensor() # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
                                                           # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    def __len__(self): # __len__ so that len(dataset) returns the size of the dataset.
        return len(self.imglists)
    
    def __getitem__(self, i): # __getitem__ to support the indexing such that dataset[i] can be used to get ith sample
        # 1. load image and kps
        image = cv2.imread(self.imglists[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,c = image.shape

        kps_path = self.imglists[i][:-4]+'.pts'
        kps = loadFromPts(kps_path)
        
        # 2. data augmentation
        if self.phase == 'train':
            # rotate
            angle = random.randint(-self.r, self.r)
            r_kps = rotatepoints(kps,[w/2,h/2],angle) # 逆时针旋转angle度
            
            # norm kps to [0,res] 
            #旋转之后的kps的方框
            left = np.min(r_kps[:, 0]) # 所有行的第0列
            right = np.max(r_kps[:, 0])
            top = np.min(r_kps[:, 1]) # 所有行的第1列
            bot = np.max(r_kps[:, 1])
            
            r_kps -=[left, top] # 坐标转换到[0,-]
            r_kps[:,0] *= self.res/(right-left) # 坐标转换到[0,res]
            r_kps[:,1] *= self.res/(bot-top)
            
            # scale
            s = random.uniform(0.9, 1.2) # uniform()方法将随机生成浮点数，它在 [x, y) 范围内
            # make scale around center 
            dx = (1-s)*self.res * 0.5 # res*0.5-s*res*0.5 缩放前的中心-缩放后的中心
            s_kps = r_kps*s + [dx,dx]
            
            # translation
            dx = random.uniform(-self.res*0.1, self.res*0.1)
            dy = random.uniform(-self.res*0.1, self.res*0.1)
            t_kps = s_kps + [dx,dy]
            
            # procrustes analysis 从两组关键点间分析出变换矩阵用于图像的变换
            d, Z, tform = procrustes(t_kps, kps) # a dict specifying the rotation, translation and scaling that maps X --> Y
            M = np.zeros([2,3],dtype=np.float32)
            M[:2,:2] = tform['rotation'].T * tform['scale']
            M[:,2] = tform['translation']
            img = cv2.warpAffine(image,M,(self.res,self.res)) # 仿射变换 将图像按照关键点变换
            new_kps = np.dot(kps,tform['rotation']) * tform['scale'] + tform['translation']

            
        
        else:
            # enlarge box 
            box = get_gtbox(kps)
            box = enlarge_box(box,0.05)
            xmin, ymin, xmax, ymax = box

            
            src = np.array([[xmin,ymin],[xmin,ymax],[xmax,ymin],[xmax,ymax]])
            dst = np.array([[0,0],[0,self.res-1],[self.res-1,0],[self.res-1,self.res-1]])
            
            # procrustes analysis
            d, Z, tform = procrustes(dst, src)
            M = np.zeros([2,3],dtype=np.float32)
            M[:2,:2] = tform['rotation'].T * tform['scale']
            M[:,2] = tform['translation']
            img = cv2.warpAffine(image,M,(self.res,self.res))

            new_kps = np.dot(kps,tform['rotation']) * tform['scale'] + tform['translation']


        if self.phase == 'train':
            # flip
            if random.random() > 0.5:
                img = img[:, ::-1] # 左右翻转
                new_kps = flippoints(new_kps, self.res)
            
            # resize
            if random.random() > 0.8:
                new_res = int(self.res*0.75)
                img = cv2.resize(img,(new_res,new_res))
                img = cv2.resize(img,(self.res,self.res))
                
        if self.target_type == 'heatmap':
            num_points = kps.shape[0]
            new_kps = new_kps.astype(np.int32)
            target = np.zeros([num_points,self.res,self.res])
            for n in range(num_points):
                target[n] = draw_gaussian(target[n], new_kps[n], sigma=self.gamma) # 构造训练的heatmap的标签
            target = torch.from_numpy(target).float() # 将numpy格式转换成torch.tensor格式
        else:
            target = torch.from_numpy(new_kps).float() # 回归landmark



        # img to tensor
        img = self.transform(img.copy()) # transforms.ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to 
                                        #a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        if self.phase == 'train':
            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
        
        
        
        return img, target, torch.from_numpy(new_kps), tform
