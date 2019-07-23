import os
import argparse
import torch.utils.data as data
from solver import Solver
from torch.backends import cudnn
from dataset import WFLW_Dataset
 

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    imgdirs_train = ['Data/afw/', 'Data/helen/trainset/', 'Data/lfpw/trainset/']
    imgdirs_test_commomset = ['Data/helen/testset/','Data/lfpw/testset/']

    # Dataset and Dataloader
    if config.phase == 'test':
        trainset=None
        train_loader = None
    else:
        trainset = WFLW_Dataset(imgdirs_train, config.phase, 'train', config.rotFactor, config.res, config.gamma)
        train_loader = data.DataLoader(trainset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers,
                                       pin_memory=True)
    testset = WFLW_Dataset(imgdirs_test_commomset, 'test', config.attr, config.rotFactor, config.res, config.gamma)
    test_loader = data.DataLoader(testset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  pin_memory=True)
    
    # Solver for training and testing.
    solver = Solver(train_loader, test_loader, config)
    if config.phase == 'train':
        solver.train()
    else:
        solver.load_state_dict(config.best_model)
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--nPoints', type=int, default=68, help='keypoint nums')
    parser.add_argument('--batch_size', type=int, default=10, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--weightDecay', type=float, default=0, help='weight decay')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--attr', type=str, default='test', help='test, pose, blur, occlusion etc')
    parser.add_argument('--gamma', type=int, default=3, help='gaussian kernel')
   
    # Augumentation options 
    parser.add_argument('--rotFactor', type=int, default=30, help='rotation factor (in degrees)') 
    parser.add_argument('--res', type=int, default=128, help='input resolution') 
    
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--best_model', default='checkpoint/models/best_checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to save best checkpoint (default: checkpoint)')

    # Directories.
    parser.add_argument('--log_dir', type=str, default='checkpoint/logs')
    parser.add_argument('--model_save_dir', type=str, default='checkpoint/models')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=2000)
    parser.add_argument('--lr_update_step', type=int, default=40000)
    
    config = parser.parse_args()
    main(config)
 