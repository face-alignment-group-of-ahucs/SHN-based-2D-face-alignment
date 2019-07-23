#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 19:07:02 2018

@author: xiang
"""
 

import torch
import os
import time
import datetime
import numpy as np
#from resnet import resnet
from utils import rmse_batch, flip_channels, shuffle_channels_for_horizontal_flipping
from model import FAN

class Solver(object):
    """Solver for training fan."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Training configurations.
        self.nPoints = config.nPoints
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.lr = config.lr
        self.weightDecay = config.weightDecay
        self.resume_iters = config.resume_iters
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.phase = config.phase
        
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_error = 1
        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        
        self.test_step = 0
        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
        self.best = 0

    def build_model(self):
        """Create network."""
        self.model = FAN(3,68)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2], weight_decay=self.weightDecay)
        self.critertion = torch.nn.MSELoss()
        self.print_network(self.model, 'model')
        self.model.to(self.device)
        

    def print_network(self, model, name):
        """Print out the network information."""
        #print(name)
        #print(model)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    def load_state_dict(self, path_best_model):
        self.model.load_state_dict(torch.load(path_best_model))
        
    
    def save_checkpoint(self, model_path, resume_iters):
        state = {
                'resume_iters': resume_iters,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'lr': self.lr,
                'min_error':self.min_error
                }

        torch.save(state, model_path)
        
    def restore_model(self):
        """Restore the trained model."""
        print('Loading the pretrained models.')
        model_path = os.path.join(self.model_save_dir, 'Checkpoint.pth.tar')
        state = torch.load(model_path)
        self.resume_iters = state['resume_iters']
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.min_error = state['min_error']
        lr = state['lr']
        self.update_lr(lr)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Decay learning rate."""
        for param_group in self.optimizer.param_groups: # The learnable parameters of a model are returned by net.parameters()
            param_group['lr'] = lr
        self.lr = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()
    
    def train(self):
        """Train network."""

        data_iter = iter(self.train_loader)
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            self.restore_model()
            start_iters = self.resume_iters
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            self.model.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            # Fetch real images and labels.
            try:
                images, targets, kps, tforms = next(data_iter)
            except:
                data_iter = iter(self.train_loader)
                images, targets, kps, tforms = next(data_iter)
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # =================================================================================== #
            #                             2. Train network                                        #
            # =================================================================================== #

            out = self.model(images)
            loss = self.critertion(out,targets)

            self.reset_grad()
            loss.backward()
            self.optimizer.step()
            # Logging.
            losses = {}
            losses['train_loss'] = loss.item()
            rmse = rmse_batch(out.cpu(), kps, tforms) # rmse: N numpy 
            losses['train_rmse'] = np.mean(rmse)
            

            # =================================================================================== #
            #                             3. Miscellaneous                                        #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in losses.items(): # 返回可遍历的(键, 值),losses有两个键值对
                    log += ", {}: {:.5f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in losses.items():
                        self.logger.scalar_summary(tag, value, i+1)



            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                model_save_path = os.path.join(self.model_save_dir, 'Checkpoint.pth.tar')
                self.save_checkpoint(model_save_path, i+1)
                print('Save model checkpoint into {}...'.format(self.model_save_dir))
                self.test()
            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0:
                lr = self.lr * 0.5
                self.update_lr(lr)
                print ('Decayed learning rates, lr: {}.'.format(lr))
        
    def test(self):
        self.model.eval()
        with torch.no_grad():
            # Start training.
            print('Start testing...')
            start_time = time.time()
            idx = 0
            Rmse = np.zeros([self.test_loader.dataset.__len__()])
            for i,(images, targets, kps, tforms) in enumerate(self.test_loader):

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                images = images.to(self.device)
#                targets = targets.to(self.device)
                bs = images.size(0)
                # =================================================================================== #
                #                             2. test network                                         #
                # =================================================================================== #
                out1 = self.model(images)
                # flip
                images_flip = torch.from_numpy(images.cpu().numpy()[:, :, :, ::-1].copy()) # 左右翻转
                images_flip = images_flip.to(self.device)
                out2 = self.model(images_flip)
                out2 = flip_channels(out2.cpu())
                out2 = shuffle_channels_for_horizontal_flipping(out2)
                out = (out1.cpu() + out2)/2
                loss = self.critertion(out, targets)


                
                # Logging
                losses = {}
                losses['test_loss'] = loss.item()
                rmse = rmse_batch(out, kps, tforms)
                Rmse[idx:idx+bs] = rmse
                idx += bs
                losses['test_rmse'] = np.mean(rmse)
                # =================================================================================== #
                #                             3. Miscellaneous                                        #
                # =================================================================================== #
            
            
                # Print out testing information.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, len(self.test_loader))
                    for tag, value in losses.items():
                        log += ", {}: {:.5f}".format(tag, value)
                    print(log)
                    if self.use_tensorboard and self.phase=='train':
                        for tag, value in losses.items():
                            self.logger.scalar_summary(tag, value, self.test_step+i+1)
            
            mean_rmse = np.mean(Rmse)

            print('Test Inter-pupil Normalisation: {}'.format(mean_rmse))
            
            # save best checkpoint
            if mean_rmse < self.min_error:
                self.min_error = mean_rmse
                print('Test Inter-pupil Normalisation Best: {}'.format(mean_rmse))
                model_save_path = os.path.join(self.model_save_dir, 'best_checkpoint.pth.tar')
                print('Save best checkpoint into {}...'.format(self.model_save_dir))
                torch.save(self.model.state_dict(), model_save_path)
                
            self.test_step += len(self.test_loader)
