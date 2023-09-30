import argparse
import os
import torch
import numpy
import torch.nn as nn
from torch.utils import data
from dataset import get_loader
from cnn_regressor import CNNRegressor
from quality_predictor_dataloader import ImageDataTestNOMOS
from data_dic import data_list
from solver import Solver
import time
def get_test_info(config):
    if config.sal_mode == 'NJU2K':
        image_root = '../testsod/NJU2K_test/NJU2K_test'
        image_source = '../testsod/NJU2K_test/NJU2K_test/test.lst'
    elif config.sal_mode == 'STERE':
        image_root = '../testsod/STERE/STERE'
        image_source = '../testsod/STERE/STERE/test.lst'
    elif config.sal_mode == 'RGBD135':
        image_root = '../testsod/RGBD135/RGBD135'
        image_source = '../testsod/RGBD135/RGBD135/test.lst'
    elif config.sal_mode == 'LFSD':
        image_root = '../testsod/LFSD/LFSD'
        image_source = '../testsod/LFSD/LFSD/test.lst'
    elif config.sal_mode == 'NLPR':
        image_root = '../testsod/NLPR/NLPR'
        image_source = '../testsod/NLPR/NLPR/test.lst'
    elif config.sal_mode == 'SIP':
        image_root = '../testsod/SIP/SIP'
        image_source = '../testsod/SIP/SIP/test.lst'
    elif config.sal_mode == 'ReDWeb-S':
        image_root = 'dataset/test/ReDWeb-S/'
        image_source = 'dataset/test/ReDWeb-S/test.lst'
    else:
        raise Exception('Invalid config.sal_mode')

    config.test_root = image_root
    config.test_list = image_source
def main(config):
    #quality prediction
    shuffle=False
    pin=True
    if config.mode == 'train':
        qp_root=config.train_root
        qp_list=config.train_list
    elif config.mode == 'test':
        qp_root=config.test_root
        qp_list=config.test_list
    qp_dataset = ImageDataTestNOMOS(qp_root, qp_list, config.qp_image_size)
    qp_loader = data.DataLoader(dataset=qp_dataset, batch_size=config.qp_batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    qp_model = CNNRegressor()
    if config.cuda:
        qp_model = qp_model.cuda()
        device = torch.device(config.device_id)
        qp_model.load_state_dict(torch.load(config.qp_pretrained_model))
    else:
        qp_model = qp_model.cpu()
        qp_model.load_state_dict(torch.load(config.qp_pretrained_model,map_location=torch.device('cpu')))
    qp_model.eval()
    # a dictionary of quality score
    qp_data={}
    with torch.no_grad():
        for image_test,(name,) in qp_loader:
            if config.cuda:
                device = torch.device(config.device_id)
                image_test= image_test.to(device)
            pred_test=qp_model(image_test)
            qp_data[name]=pred_test.cpu().numpy().item()
        qp_sorted_dict = dict(sorted(qp_data.items(), key=lambda x: x[1]))

    rgb,depth,gt,quality = data_list(qp_root,qp_list,qp_sorted_dict,config)
    if config.mode == 'train':
        train_loader = get_loader(qp_root,rgb,depth,gt,quality,config)
        print('train dataset loaded:')
        print('numper of training  samples',len(train_loader))
        if not os.path.exists("%s/demo-%s" % (config.save_folder, time.strftime("%d"))):
            os.mkdir("%s/demo-%s" % (config.save_folder, time.strftime("%d")))
        config.save_folder = "%s/demo-%s" % (config.save_folder, time.strftime("%d"))
        train = Solver(train_loader, None,config)
        train.train()
    elif config.mode == 'test':
        #get_test_info(config)
        test_loader = get_loader(qp_root,rgb,depth,gt,quality,config, mode='test')
        path = os.path.join(config.test_folder, config.sal_mode)
        if not os.path.exists(config.test_folder): os.makedirs(path)
        config.test_folder=path
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    resnet101_path = '../pretrained-pytorch/resnet101-5d3b4d8f.pth'
    resnet50_path = '../pretrained-pytorch/resnet50-19c8e357.pth'
    vgg16_path = '../pretrained-pytorch/vgg16-397923af.pth'
    conformer_path='../pretrained-pytorch/Conformer_base_patch16.pth'
    cswin_path='../pretrained-pytorch/cswin_large_384.pth'
    densenet161_path = '../pretrained-pytorch/densenet161-8d451a50.pth'
    pretrained_path = {'resnet101': resnet101_path, 'resnet50': resnet50_path, 'vgg16': vgg16_path,
                       'densenet161': densenet161_path,'conformer':conformer_path,'cswin':cswin_path}

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0000001)  # Learning rate resnet:4e-4
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=str, default='cuda:0')
    parser.add_argument('--qp_image_size', type=int, default=768)
    parser.add_argument('--qp_batch_size', type=int, default=1)
    parser.add_argument('--qp_pretrained_model', type=str, default='../diqa-sip/final.pth')
   
    # Training settings
    parser.add_argument('--arch', type=str, default='conformer'
                        , choices=['resnet', 'vgg','densenet','conformer','cswin'])  # resnet, vgg or densenet
    parser.add_argument('--pretrained_model', type=str, default=pretrained_path)  # pretrained backbone model
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)  # only support 1 now
    
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')  # pretrained JL-DCF model
    parser.add_argument('--save_folder', type=str, default='checkpoints/')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)
    parser.add_argument('--network', type=str, default='conformer'
                        , choices=['resnet50', 'resnet101', 'vgg16', 'densenet161','conformer','cswin'])  # Network Architecture
    #conformer setting
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--channel_ratio', type=int, default=6)
    parser.add_argument('--embed_dim', type=int, default=576)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=9)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    # Train data
    parser.add_argument('--train_root', type=str, default='../RGBDcollection')
    parser.add_argument('--train_list', type=str, default='../RGBDcollection/train.lst')
    parser.add_argument('--img_folder', type=str, default='../DUTS-TR/DUTS-TR-Image')
    parser.add_argument('--gt_folder', type=str, default='../DUTS-TR/DUTS-TR-Mask')
    

    # Testing settings
    parser.add_argument('--model_t', type=str, default='./checkpoints/demo-08/epoch_40.pth')  # Snapshot
    parser.add_argument('--model', type=str, default='./checkpoints/demo-08/epoch_40.pth')  # Snapshot
    parser.add_argument('--test_folder', type=str, default='testint')  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='RGBD135',
                        choices=['NJU2K', 'NLPR', 'STERE', 'RGBD135', 'LFSD', 'SIP', 'ReDWeb-S'])  # Test image dataset
    parser.add_argument('--test_root', type=str, default='../testsod/RGBD135/RGBD135')
    parser.add_argument('--test_list', type=str, default='../testsod/RGBD135/RGBD135/test.lst')
    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    #get_test_info(config)

    main(config)
