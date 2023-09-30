import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
from timm.models.layers import DropPath, trunc_normal_
import os
import cv2
import numpy
import numpy as np
k_channels=[384,384,384,768,1536]
d_channels=576
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,q,k,v


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_norm=self.norm1(x)
        x_att,q,k,v=self.attn(x_norm)
        x = x + self.drop_path(x_att)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,x_att,q,k,v


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)
        #print('cnn_block',x.shape,x2.shape)
        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)
        #print('FCdown_block',x_st.shape)
        x_t,x_att,q,k,v = self.trans_block(x_st + x_t)
        #print('tran_block',x_t.shape,q.shape)
        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        #print('FCUP_block',x_t_r.shape)
        x = self.fusion_block(x, x_t_r, return_x_2=False)
        #print('fusion_block',x.shape)

        return x,x_att, x_t,q,k,v


class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, last_fusion=last_fusion
                    )
            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x,y):
        #B = x.shape[0]
        B = y.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        conv_features=[]
        tran_features=[]
        q=[]
        k=[]
        v=[]
        x_att=[]
        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        y_base = self.maxpool(self.act1(self.bn1(self.conv1(y))))
        #print('x_base',x_base.shape)
        conv_features.append(x_base)
        tran_features.append(y_base)
        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)
        conv_features.append(x)

        y_t = self.trans_patch_conv(y_base).flatten(2).transpose(1, 2)
        #print('x_t flatten',x_t.shape)
        tran_features.append(y_t)
       
        y_t = torch.cat([cls_tokens, y_t], dim=1)
        #print('y_t n tokens',y_t.shape)
        y_t,x_att1,q1,k1,v1 = self.trans_1(y_t)
        #print('y_t tran_1 q k  v',y_t.shape,q1.shape,k1.shape,v1.shape)
        tran_features.append(y_t)
        q.append(q1)
        k.append(k1)
        v.append(v1)
        x_att.append(x_att1)
        # 2 ~ final 
        for i in range(2, self.fin_stage):
            x, x_atti,y_t,qi,ki,vi = eval('self.conv_trans_' + str(i))(x, y_t)
            conv_features.append(x)
            tran_features.append(y_t)
            q.append(qi)
            k.append(ki)
            v.append(vi)
            x_att.append(x_atti)
        
        return conv_features,tran_features

class RGBDInModule(nn.Module):
    def __init__(self, backbone):
        super(RGBDInModule, self).__init__()
        self.backbone = backbone
     
    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        
    def forward(self, x,y):
        conv,tran = self.backbone(x,y)
        '''for i in range(len(conv)):
            print(i,"     ",conv[i].shape,tran[i].shape)'''
        return conv,tran # list of tensor that compress model output
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        avg_pool = self.avg_pool(x).view(batch_size, num_channels)
        fc1 = self.relu(self.fc1(avg_pool))
        fc2 = self.sigmoid(self.fc2(fc1))
        fc2 = fc2.view(batch_size, num_channels, 1, 1)
        out=fc2*x
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return attention * x
    
class RGBD_encoder(nn.Module):
    def __init__(self):
        super(RGBD_encoder, self).__init__()
        
        
        self.relu = nn.ReLU(inplace=True)
        self.conv_stage1=SpatialAttention(k_channels[0])
        self.conv_stage4=SpatialAttention(k_channels[3])
        self.conv_stage5=SpatialAttention(k_channels[4])
        self.conv_staged=SpatialAttention(d_channels)
 
        self.deconv_stage1=nn.ConvTranspose2d(k_channels[0],k_channels[0],kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.deconv_stage4=nn.ConvTranspose2d(k_channels[3],k_channels[2],kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.deconv_stage5=nn.ConvTranspose2d(k_channels[4],k_channels[3],kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.deconv_staged=nn.ConvTranspose2d(d_channels,d_channels,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
    

        self.ca_stage3=ChannelAttention(k_channels[1])
        self.ca_stage4=ChannelAttention(k_channels[2])
        self.ca_stage5=ChannelAttention(k_channels[3])
        self.ca_staged=ChannelAttention(d_channels)

        self.sal_stage3=nn.ConvTranspose2d(k_channels[2],1,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.sal_stage4=nn.ConvTranspose2d(k_channels[2],k_channels[1],kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.sal_stage5=nn.ConvTranspose2d(k_channels[3],k_channels[2],kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.sal_staged=nn.ConvTranspose2d(d_channels,1,kernel_size=4, stride=8, padding=1, output_padding=3, dilation=2)      

        
    def forward(self, feat_rgb,modality):
        Fout=[]
        if modality=='depth':
            #spatial attention
            f_branch1 = self.conv_staged(feat_rgb[0])
            f_branch2 = self.conv_staged(feat_rgb[1])
            f_branch3 = self.conv_staged(feat_rgb[2])
            f_branch4 = self.conv_staged(feat_rgb[3])
            f_branch5 = self.conv_staged(feat_rgb[4])
            #concatenation of adjacent features
            f_out1 = self.deconv_staged(f_branch1)
            f_out2 = self.deconv_staged(f_branch2)
            f_out3 = self.deconv_staged(f_branch3)
            f_out3 = f_out3+f_out2+f_out1
            f_out4 = self.deconv_staged(f_branch4)
            f_out5 = self.deconv_staged(f_branch5)
            #channel attention
            f_out3ca = self.ca_staged(f_out3)
            Fout.append(f_out3ca)
            f_out4ca = self.ca_staged(f_out4)
            Fout.append(f_out4ca)
            f_out5ca = self.ca_staged(f_out5)
            Fout.append(f_out5ca)
        
            #saliency maps at multi scales
            f_sal_3=self.sal_staged(f_out5ca+f_out4ca+f_out3ca)
        else: 

            #spatial attention
            f_branch1 = self.conv_stage1(feat_rgb[0])
            f_branch2 = self.conv_stage1(feat_rgb[1])
            f_branch3 = self.conv_stage1(feat_rgb[2])
            f_branch4 = self.conv_stage4(feat_rgb[3])
            f_branch5 = self.conv_stage5(feat_rgb[4])
            #concatenation of adjacent features
            f_out1 = self.deconv_stage1(f_branch1)
            f_out2 = self.deconv_stage1(f_branch2)
            f_out3 = self.deconv_stage1(f_branch3)+f_out2+f_out1
            f_out4 = self.deconv_stage4(f_branch4)
            f_out5 = self.deconv_stage5(f_branch5)
            #channel attention
            f_out3ca = self.ca_stage3(f_out3)
            Fout.append(f_out3ca)
            f_out4ca = self.ca_stage4(f_out4)
            Fout.append(f_out4ca)
            f_out5ca = self.ca_stage5(f_out5)
            Fout.append(f_out5ca)
            #saliency maps at multi scales
            f_sal_5=self.sal_stage5(f_out5ca)+f_out4ca
            f_sal_4=self.sal_stage4(f_sal_5)+f_out3ca
            f_sal_3=self.sal_stage3(f_sal_4)

      

        
        '''print(f_branch1.shape,f_out1.shape)
        print(f_branch2.shape,f_out2.shape)
        print(f_branch3.shape,f_out3.shape,f_out3ca.shape,f_sal_3.shape)
        print(f_branch4.shape,f_out4.shape,f_out4ca.shape)
        print(f_branch5.shape,f_out5.shape,f_out5ca.shape)'''
  

        
        return f_sal_3,Fout

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
       
#modality specific representation
class MSR(nn.Module):
    def __init__(self,in_features,h_features,  out_features):
        super(MSR, self).__init__() 
        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = MLP(in_features, h_features, out_features)
        
    def forward(self, x):
        GapOut = self.GAP(x)
        #print(GapOut.shape)
        GapOut = GapOut.view(GapOut.size(0), -1)
        #print(GapOut.shape)
        gate = self.mlp(GapOut)
        
        return gate


class ShuffleChannelAttention(nn.Module):
    def __init__(self, channel=64,reduction=16,kernel_size=3,groups=8):
        super(ShuffleChannelAttention, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.g=groups
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,3,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        
    
    def forward(self, x) :
        b,c,h,w=x.shape
        residual=x
        max_result=self.maxpool(x)
        #print('***Shuffle chaneel***')
        #print('max',max_result.shape)
        shuffled_in=max_result.view(b,self.g,c//self.g,1,1).permute(0,2,1,3,4).reshape(b,c,1,1)
        #print('shuffled',shuffled_in.shape)
        max_out=self.se(shuffled_in)
        #print('se',max_out.shape)
        output1=self.sigmoid(max_out)
        output1=output1.view(b,c,1,1)
        #print('output1',output1.shape)
        output2=self.sigmoid(max_result)
        output=output1+output2
        return (output*x)+residual    

class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x 

class RGBD_SOD(nn.Module):
    def __init__(self,RGBDInModule,RGBD_encoder):
        super(RGBD_SOD, self).__init__() 
        self.relu = nn.ReLU(inplace=True)
        self.RGBDInModule = RGBDInModule
        self.encoder = RGBD_encoder
        
        self.msr1 = MSR(in_features=384, h_features=128, out_features=2)
        self.msr2 = MSR(in_features=768, h_features=128,  out_features=2)
        self.msr3 = MSR(in_features=576, h_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=1)
        self.ca1 = ShuffleChannelAttention(channel=768,reduction=16,kernel_size=3,groups=4)
        self.ca2 = ShuffleChannelAttention(channel=1536,reduction=32,kernel_size=3,groups=8)
        
        self.sa = SpatialGroupEnhance(groups=4)
        self.upf1 = nn.ConvTranspose2d(576,384,kernel_size=3, stride=4, padding=0, output_padding=1, dilation=1)
        self.upf2 = nn.ConvTranspose2d(576,384,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.upf3 = nn.Conv2d(576,768,1,1)
        self.last_conv = nn.Conv2d(64,1, 1, 1)
        self.f_up3 = nn.ConvTranspose2d(1536,768,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.f_up2 = nn.ConvTranspose2d(768,768,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.f_up1 = nn.ConvTranspose2d(768,64,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)       

    def forward(self, rgb,depth):
        feat_rgb,feat_depth_r  = self.RGBDInModule(rgb,depth)
         
        feat_depth=[]
        for j in range(2,len(feat_depth_r)):
            B,_,C=feat_depth_r[j].shape
            x=feat_depth_r[j]
            
            feat_depth.append(x[:, 1:].transpose(1, 2).reshape(B, C, 24,24))
        feat_rgb_sel=[]
        feat_depth_sel=[]
        indices_to_operate = [2, 3, 4, 8, 11]
        for index in indices_to_operate:
            feat_rgb_sel.append(feat_rgb[index])
            feat_depth_sel.append(feat_depth[index])
        '''for i in range(len(feat_depth_sel)):
            print(i,feat_rgb_sel[i].shape,feat_depth_sel[i].shape)'''
        rgb_enc,Frgb_out = self.encoder(feat_rgb_sel,'rgb')
        depth_enc,Fdepth_out = self.encoder(feat_depth_sel,'depth')
        '''for i in range(len(Frgb_out)):
            print(i,'frgb_out',Frgb_out[i].shape,'fdepth',Fdepth_out[i].shape)'''
      
        rgb_param1 = self.msr1(Frgb_out[0])
        #print(rgb_param1[:,0].shape)
        rgb_param2 = self.msr1(Frgb_out[1])
        #print(rgb_param2.shape)
        rgb_param3 = self.msr2(Frgb_out[2])
        #print(rgb_param3.shape)
        depth_param1 = self.msr3(Fdepth_out[0])
        depth_param2 = self.msr3(Fdepth_out[1])
        depth_param3 = self.msr3(Fdepth_out[2])


        corr_rgb2d1 = self.softmax(rgb_param1[:,0]*Fdepth_out[0] +rgb_param1[:,1])
        corr_rgb2d2 = self.softmax(rgb_param2[:,0]*Fdepth_out[1] +rgb_param2[:,1])
        corr_rgb2d3 = self.softmax(rgb_param3[:,0]*Fdepth_out[2] +rgb_param3[:,1])

        corr_d2rgb1 = self.softmax(depth_param1[:,0]*Frgb_out[0]+depth_param1[:,1])
        corr_d2rgb2 = self.softmax(depth_param2[:,0]*Frgb_out[1]+depth_param2[:,1])
        corr_d2rgb3 = self.softmax(depth_param3[:,0]*Frgb_out[2]+depth_param3[:,1])

      
        fusion3 = torch.cat((self.upf3(corr_rgb2d3),corr_d2rgb3),dim=1)
        #up_fusion3=self.up(fusion3 )
        fusion2 = torch.cat((self.upf2(corr_rgb2d2),corr_d2rgb2),dim=1)
        #up_fusion2=self.up(fusion2 )
        fusion1 = torch.cat((self.upf1(corr_rgb2d1),corr_d2rgb1),dim=1)

        '''print(fusion1.shape)
        print(fusion2.shape)
        print(fusion3.shape)'''
       

        fusion_ca3 = self.ca2(fusion3)
        fusion_sa3 = self.sa(fusion3)
        f_att3 = self.f_up3((fusion_ca3 + fusion_sa3))
        #print('afeter ca3', fusion_ca3.shape,fusion_sa3.shape,f_att3.shape)
        fusion_ca2 = self.ca1(fusion2)
        fusion_sa2 = self.sa(fusion2)
        f_att2 = self.f_up2((fusion_ca2+ fusion_sa2)+f_att3)
        #print('afeter ca2', fusion_ca2.shape,fusion_sa2.shape,f_att2.shape)
        fusion_ca1 = self.ca1(fusion1)
        fusion_sa1 = self.sa(fusion1)
        f_att1 = self.f_up1((fusion_ca1+ fusion_sa1)+f_att2)
        '''print('afeter ca1', fusion_ca1.shape,fusion_sa1.shape,f_att1.shape)
        print(rgb_enc.shape)
        print(depth_enc.shape)'''
        
        final = self.last_conv(f_att1)
        #print(final.shape)
        return final,rgb_enc,depth_enc
        
def build_model(network='conformer', base_model_cfg='conformer'):
    backbone = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True) 
    return RGBD_SOD(RGBDInModule(backbone),RGBD_encoder())
