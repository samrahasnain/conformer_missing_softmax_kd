import torch
from torch.nn import functional as F
from RGBDincomplete import build_model
from RGBDincomplete_student import build_model_s
from distiller import build_model_kd
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse1 = (192,192)
size_coarse2 = (96,96)
size_coarse3 = (48,48)
size_coarse4 = (24,24)
from tqdm import trange, tqdm
import torch
from torch.autograd import profiler




class Solver(object):
    def __init__(self, train_loader,test_loader, config):
        self.train_loader = train_loader
     
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net_t = build_model(self.config.network, self.config.arch)
        self.net_s = build_model_s(self.config.network, self.config.arch)
        self.net_kd=build_model_kd(self.net_t, self.net_s)
        #self.net.eval()
        print('Loading pre-trained teacher model for kd from %s...' % self.config.model_t)
        self.net_t.load_state_dict(torch.load(self.config.model_t))
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net_kd.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                self.net_s.RGBDInModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
                # load pretrained backbone
            else:
                print('Loading pretrained model to resume training')
                self.net_kd.load_state_dict(torch.load(self.config.load))  # load pretrained model
        
        if self.config.cuda:
            self.net_t = self.net_t.cuda()
            self.net_s = self.net_s.cuda()
            self.net_kd = self.net_kd.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam([{'params': self.net_s.parameters()}, {'params': self.net_kd.Connectors.parameters()}], lr=self.lr, weight_decay=self.wd)
        print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in net_t.parameters()])))
        print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in net_s.parameters()])))
        self.print_network(self.net, 'Incomplete modality RGBD SOD Structure')

    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        param_size = 0

        for p in model.parameters():
            param_size += p.nelement() * p.element_size()
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        #print(name)
        #print(model)
        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.6f}MB'.format(size_all_mb))
        print("The number of trainable parameters: {:.6f}".format(num_params_t))
        print("The number of parameters: {:.6f}".format(num_params))

    # build the network
    '''def build_model(self):
        self.net = build_model(self.config.network, self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')'''

    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                preds,rgb_enc,depth_enc = self.net_kd(images,depth)
                #preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_sod.png')
                cv2.imwrite(filename, multi_fuse)
                
  

        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        step=0  
        loss_vals=  []
     
        aveGrad = 0
        
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in tqdm(enumerate(self.train_loader)):
                sal_image, sal_depth, sal_label,name = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                        'sal_label'],data_batch['name'][0]

                
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label = sal_image.to(device), sal_depth.to(device), sal_label.to(device)
                
                step+=1
                self.optimizer.zero_grad()
                sal_final,rgb_en1,depth_en1,loss_distill= self.net_kd(sal_image,sal_depth)
                

                sal_loss_final =  F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                sal_loss_rgb =  F.binary_cross_entropy_with_logits(rgb_en1, sal_label, reduction='sum')
                sal_loss_depth =  F.binary_cross_entropy_with_logits(depth_en1, sal_label, reduction='sum')
                    
                sal_rgb_only_loss = sal_loss_final + sal_loss_rgb/2 + sal_loss_depth/2+loss_distill
                r_sal_loss += sal_rgb_only_loss.data
                r_sal_loss_item+=sal_rgb_only_loss.item() * sal_image.size(0)
                sal_rgb_only_loss.backward()

                self.optimizer.step()


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_kd.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            



            loss_vals.append(train_loss)
                
            print('Epoch:[%2d/%2d] | Train Loss : %.3f | Learning rate : %0.7f' % (epoch, self.config.epoch,train_loss,self.optimizer.param_groups[0]['lr']))
                
                
        
            # save model
        torch.save(self.net_kd.state_dict(), '%s/final.pth' % self.config.save_folder)
            
    

            
