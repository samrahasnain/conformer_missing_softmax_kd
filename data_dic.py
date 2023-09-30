import numpy as np
def data_list(qp_root,qp_list,qp_sorted_dict,config):
  rgb=[]
  depth=[]
  gt=[]
  quality=[]
  if config.mode=='train':
    index=-8
    im_ext='_ori.jpg'
    de_ext='_Depth.png'
    gt_ext='_GT.png'
  elif config.mode == 'test':
    index=-4
    if config.sal_mode == 'LFSD' or config.sal_mode == 'SIP' or config.sal_mode == 'STERE' :
        im_ext='.jpg'
        de_ext='.png'
        gt_ext='_GT.png'
    elif config.sal_mode == 'RGBD135':
        im_ext='.jpg'
        de_ext='.bmp'
        gt_ext='_GT.png'
    elif config.sal_mode == 'NLPR' or config.sal_mode == 'NJU2K':
        im_ext='.jpg'
        de_ext='_Depth.bmp'
        gt_ext='_GT.png'
  with open(qp_list, 'r') as file:
      line = file.readline()
  im_name = line.split()[0].split('/')[0]
  de_name = line.split()[1].split('/')[0]
  gt_name = 'GT'
 

  for key, value in qp_sorted_dict.items():
      if value > 0.05:
          rgb.append((im_name+'/'+key[:index]+im_ext))
          depth.append((de_name+'/'+key[:index]+de_ext))
          gt.append((gt_name+'/'+key[:index]+gt_ext))
          quality.append(1)
      else:
          rgb.append((im_name+'/'+key[:index]+im_ext))
          depth.append((de_name+'/'+key[:index]+de_ext))
          gt.append((gt_name+'/'+key[:index]+gt_ext))
          quality.append(0)

  return rgb,depth,gt,quality
