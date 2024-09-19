# # -*- coding: utf-8 -*-

# from __future__ import print_function, division

# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import time
# import os
# import scipy.io
# import yaml
# import math
# from tqdm import tqdm
# from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
# from utils import fuse_all_conv_bn
# version =  torch.__version__
# #fp16
# # try:
# #     from apex.fp16_utils import *
# # except ImportError: # will be 3.x series
# #     print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

# ######################################################################
# # Options
# # --------

# parser = argparse.ArgumentParser(description='Test')
# parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
# parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
# parser.add_argument('--test_dir',default='../Market1501',type=str, help='./test_data')
# parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
# parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
# parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
# parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
# parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
# parser.add_argument('--use_hr', action='store_true', help='use hr18 net' )
# parser.add_argument('--PCB', action='store_true', help='use PCB' )
# parser.add_argument('--multi', action='store_true', help='use multiple query' )
# parser.add_argument('--fp16', action='store_true', help='use fp16.' )
# parser.add_argument('--ibn', action='store_true', help='use ibn.' )
# parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

# opt = parser.parse_args()
# ###load config###
# # load the training config
# config_path = os.path.join('./model',opt.name,'opts.yaml')
# with open(config_path, 'r') as stream:
#         config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
# opt.fp16 = config['fp16'] 
# opt.PCB = config['PCB']
# opt.use_dense = config['use_dense']
# # opt.use_NAS = config['use_NAS']
# opt.stride = config['stride']
# if 'use_swin' in config:
#     opt.use_swin = config['use_swin']
# if 'use_swinv2' in config:
#     opt.use_swinv2 = config['use_swinv2']
# if 'use_convnext' in config:
#     opt.use_convnext = config['use_convnext']
# if 'use_efficient' in config:
#     opt.use_efficient = config['use_efficient']
# if 'use_hr' in config:
#     opt.use_hr = config['use_hr']

# if 'nclasses' in config: # tp compatible with old config files
#     opt.nclasses = config['nclasses']
# else: 
#     opt.nclasses = 751 

# if 'ibn' in config:
#     opt.ibn = config['ibn']
# if 'linear_num' in config:
#     opt.linear_num = config['linear_num']

# str_ids = opt.gpu_ids.split(',')
# #which_epoch = opt.which_epoch
# name = opt.name
# test_dir = opt.test_dir

# gpu_ids = []
# for str_id in str_ids:
#     id = int(str_id)
#     if id >=0:
#         gpu_ids.append(id)

# print('We use the scale: %s'%opt.ms)
# str_ms = opt.ms.split(',')
# ms = []
# for s in str_ms:
#     s_f = float(s)
#     ms.append(math.sqrt(s_f))

# # set gpu ids
# # if len(gpu_ids)>0:
# #     torch.cuda.set_device(gpu_ids[0])
# #     cudnn.benchmark = True
# device = torch.device("cpu")

# ######################################################################
# # Load Data
# # ---------
# #
# # We will use torchvision and torch.utils.data packages for loading the
# # data.
# #
# if opt.use_swin:
#     h, w = 224, 224
# else:
#     h, w = 256, 128

# data_transforms = transforms.Compose([
#         transforms.Resize((h, w), interpolation=3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ############### Ten Crop        
#         #transforms.TenCrop(224),
#         #transforms.Lambda(lambda crops: torch.stack(
#          #   [transforms.ToTensor()(crop) 
#           #      for crop in crops]
#            # )),
#         #transforms.Lambda(lambda crops: torch.stack(
#          #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
#           #       for crop in crops]
#           # ))
# ])

# if opt.PCB:
#     data_transforms = transforms.Compose([
#         transforms.Resize((384,192), interpolation=3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
#     ])
#     h, w = 384, 192


# data_dir = test_dir

# if opt.multi:
#     image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                              shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
# else:
#     image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                              shuffle=False, num_workers=16) for x in ['gallery','query']}
# class_names = image_datasets['query'].classes
# # use_gpu = torch.cuda.is_available()

# ######################################################################
# # Load model
# #---------------------------
# def load_network(network):
#     save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
#     try:
#         network.load_state_dict(torch.load(save_path))
#     except: 
#         if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
#             print("Compiling model...")
#             # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
#             torch.set_float32_matmul_precision('high')
#             network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
#         network.load_state_dict(torch.load(save_path))

#     return network


# ######################################################################
# # Extract feature
# # ----------------------
# #
# # Extract feature from  a trained model.
# #
# def fliplr(img):
#     '''flip horizontal'''
#     inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
#     img_flip = img.index_select(3,inv_idx)
#     return img_flip

# def extract_feature(model,dataloaders):
#     #features = torch.FloatTensor()
#     # count = 0
#     pbar = tqdm()
#     if opt.linear_num <= 0:
#         if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
#             opt.linear_num = 1024
#         elif opt.use_efficient:
#             opt.linear_num = 1792
#         elif opt.use_NAS:
#             opt.linear_num = 4032
#         else:
#             opt.linear_num = 2048

#     for iter, data in enumerate(dataloaders):
#         img, label = data
#         n, c, h, w = img.size()
#         # count += n
#         # print(count)
#         pbar.update(n)
#         ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

#         if opt.PCB:
#             ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

#         for i in range(2):
#             if(i==1):
#                 img = fliplr(img)
#             input_img = Variable(img.cuda())
#             for scale in ms:
#                 if scale != 1:
#                     # bicubic is only  available in pytorch>= 1.1
#                     input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
#                 outputs = model(input_img) 
#                 ff += outputs
#         # norm feature
#         if opt.PCB:
#             # feature size (n,2048,6)
#             # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
#             # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
#             fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
#             ff = ff.div(fnorm.expand_as(ff))
#             ff = ff.view(ff.size(0), -1)
#         else:
#             fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
#             ff = ff.div(fnorm.expand_as(ff))

        
#         if iter == 0:
#             features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
#         #features = torch.cat((features,ff.data.cpu()), 0)
#         start = iter*opt.batchsize
#         end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
#         features[ start:end, :] = ff
#     pbar.close()
#     return features

# def get_id(img_path):
#     camera_id = []
#     labels = []
#     for path, v in img_path:
#         #filename = path.split('/')[-1]
#         filename = os.path.basename(path)
#         label = filename[0:4]
#         camera = filename.split('c')[1]
#         if label[0:2]=='-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#         camera_id.append(int(camera[0]))
#     return camera_id, labels

# gallery_path = image_datasets['gallery'].imgs
# query_path = image_datasets['query'].imgs

# gallery_cam,gallery_label = get_id(gallery_path)
# query_cam,query_label = get_id(query_path)

# if opt.multi:
#     mquery_path = image_datasets['multi-query'].imgs
#     mquery_cam,mquery_label = get_id(mquery_path)

# ######################################################################
# # Load Collected data Trained model
# print('-------test-----------')
# if opt.use_dense:
#     model_structure = ft_net_dense(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
# elif opt.use_NAS:
#     model_structure = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
# elif opt.use_swin:
#     model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
# elif opt.use_swinv2:
#     model_structure = ft_net_swinv2(opt.nclasses, (h,w),  linear_num=opt.linear_num)
# elif opt.use_convnext:
#     model_structure = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
# elif opt.use_efficient:
#     model_structure = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
# elif opt.use_hr:
#     model_structure = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
# else:
#     model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)

# if opt.PCB:
#     model_structure = PCB(opt.nclasses)

# #if opt.fp16:
# #    model_structure = network_to_half(model_structure)


# model = load_network(model_structure)

# # Remove the final fc layer and classifier layer
# if opt.PCB:
#     #if opt.fp16:
#     #    model = PCB_test(model[1])
#     #else:
#         model = PCB_test(model)
# else:
#     #if opt.fp16:
#         #model[1].model.fc = nn.Sequential()
#         #model[1].classifier = nn.Sequential()
#     #else:
#         model.classifier.classifier = nn.Sequential()

# # Change to test mode
# model = model.eval()
# # if use_gpu:
# #     model = model.cuda()


# print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
# model = fuse_all_conv_bn(model)

# # We can optionally trace the forward method with PyTorch JIT so it runs faster.
# # To do so, we can call `.trace` on the reparamtrized module with dummy inputs
# # expected by the module.
# # Comment out this following line if you do not want to trace.
# #dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
# #model = torch.jit.trace(model, dummy_forward_input)

# print(model)
# # Extract feature
# since = time.time()
# with torch.no_grad():
#     gallery_feature = extract_feature(model,dataloaders['gallery'])
#     query_feature = extract_feature(model,dataloaders['query'])
#     if opt.multi:
#         mquery_feature = extract_feature(model,dataloaders['multi-query'])
# time_elapsed = time.time() - since
# print('Training complete in {:.0f}m {:.2f}s'.format(
#             time_elapsed // 60, time_elapsed % 60))
# # Save to Matlab for check
# result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
# scipy.io.savemat('pytorch_result.mat',result)

# print(opt.name)
# result = './model/%s/result.txt'%opt.name
# os.system('python evaluate_gpu.py | tee -a %s'%result)

# if opt.multi:
#     result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
#     scipy.io.savemat('multi_query.mat',result)




# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from tqdm import tqdm
from model import ft_net
from utils import fuse_all_conv_bn
# Options
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='Market1501/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--linear_num', default=751, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_efficient', action='store_true', help='use efficie  nt-b4')
parser.add_argument('--use_hr', action='store_true', help='use hr18 net')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')
parser.add_argument('--ibn', action='store_true', help='use ibn.')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()

# Load config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    print(config)
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.stride = config['stride']
opt.droprate = config['droprate']

if not (0 <= opt.droprate <= 1):
    raise ValueError(f"Dropout rate must be between 0 and 1, but got {opt.droprate}")

if 'use_convnext' in config:
    opt.use_convnext = config['use_convnext']
if 'use_efficient' in config:
    opt.use_efficient = config['use_efficient']
if 'use_hr' in config:
    opt.use_hr = config['use_hr']

if 'nclasses' in config:
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751

if 'ibn' in config:
    opt.ibn = config['ibn']
if 'linear_num' in config:
    opt.linear_num = config['linear_num']

# Set device to CPU
device = torch.device("cpu")

# Data loading
h, w = 256, 128
data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    h, w = 384, 192

data_dir = opt.test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query', 'multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=4) for x in ['gallery', 'query', 'multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=4) for x in ['gallery', 'query']}

class_names = image_datasets['query'].classes

# # # Load model
# def load_network(network):
#     save_path = os.path.join('./model', opt.name, 'net_%s.pth' % opt.which_epoch)
#     network.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
#     return network










def load_network(network):
    save_path = os.path.join('./model', opt.name, 'net_%s.pth' % opt.which_epoch)
    pretrained_state_dict = torch.load(save_path, map_location=device)
    
    # Print the architecture name of the pretrained model
    if 'resnet50' in save_path:
        pretrained_architecture = 'ResNet50'
    elif 'resnet18' in save_path:
        pretrained_architecture = 'ResNet18'
    else:
        pretrained_architecture = 'Unknown architecture'
    
    print(f"Pretrained model architecture: {pretrained_architecture}")
    
    # Print the keys of the pretrained state dictionary in a single line
    print("Pretrained state_dict keys:")
    print(", ".join(pretrained_state_dict.keys()))  # Print keys in a single line
    
    # Print the total number of keys in the pretrained state dictionary
    print(f"Number of keys in pretrained state_dict: {len(pretrained_state_dict.keys())}")
    
    # Print the architecture name of the current model
    current_model_architecture = network.__class__.__name__
    print(f"Current model architecture: {current_model_architecture}")
    
    # Print the keys of the model's state dictionary before loading in a single line
    print("Model state_dict keys before loading:")
    model_state_dict = network.state_dict()
    print(", ".join(model_state_dict.keys()))  # Print keys in a single line
    
    # Print the total number of keys in the model's state dictionary before loading
    print(f"Number of keys in model state_dict before loading: {len(model_state_dict.keys())}")
    
    # Filter out keys starting with 'classifier'
    filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('classifier')}
    
    # Load the filtered state dictionary into the model
    network.load_state_dict(filtered_state_dict, strict=False)
    
    # Print the keys of the model's state dictionary after loading in a single line
    print("Model state_dict keys after loading:")
    model_state_dict = network.state_dict()
    print(", ".join(model_state_dict.keys()))  # Print keys in a single line
    
    # Print the total number of keys in the model's state dictionary after loading
    print(f"Number of keys in model state_dict after loading: {len(model_state_dict.keys())}")
    
    return network




def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    pbar = tqdm()
    ms = [math.sqrt(float(s)) for s in opt.ms.split(',')]

    if opt.linear_num <= 0:
        if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
            opt.linear_num = 1024
        elif opt.use_efficient:
            opt.linear_num = 1792
        elif opt.use_NAS:
            opt.linear_num = 4032
        else:
            opt.linear_num = 2048

    # Initialize features on CPU
    features = torch.FloatTensor(len(dataloaders.dataset), opt.linear_num).zero_()

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()

        pbar.update(n)
        ff = torch.FloatTensor(n, opt.linear_num).zero_()

        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_()  # We have six parts

        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = img  # No need for Variable anymore
            for scale in ms:
                if scale != 1:
                    # Bicubic is only available in pytorch>=1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs

        # Normalize feature
        if opt.PCB:
            # Feature size (n, 2048, 6)
            # 1. To treat every part equally, calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        start = iter * opt.batchsize
        end = min((iter + 1) * opt.batchsize, len(dataloaders.dataset))
        features[start:end, :] = ff

    pbar.close()
    print(f'Processed features for {end} images')
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)

# Main code execution
if __name__ == '__main__':
    # if opt.PCB: 
    #     model_structure = PCB(opt.nclasses)
    # else:
    #     if opt.use_dense:
    #         model_structure = ft_net_dense(opt.nclasses, opt.stride)
        # elif opt.use_hr:
        #     model_structure = ft_net_hr(opt.nclasses)
        # elif opt.use_efficient:
        #     model_structure = ft_net_efficient(opt.nclasses)
        # else:
    model_structure = ft_net(opt.nclasses, opt.droprate, opt.stride)

    model = load_network(model_structure)
    model = model.to(device)                                          
    model = model.eval()
    # Print a few examples from the query and gallery sets
    print(model)
   # Feature extraction and conversion
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders['gallery'])
        query_feature = extract_feature(model, dataloaders['query'])
        gallery_feature = gallery_feature.cpu()
        query_feature = query_feature.cpu()

        # Convert to NumPy arrays
        gallery_feature_np = gallery_feature.numpy()
        query_feature_np = query_feature.numpy()

        if opt.multi:
            mquery_feature = extract_feature(model, dataloaders['multi-query'])
            mquery_feature = mquery_feature.cpu()
            mquery_feature_np = mquery_feature.numpy()

    # Prepare results
    result = {
        'gallery_f': gallery_feature_np,
        'gallery_label': gallery_label,
        'gallery_cam': gallery_cam,
        'query_f': query_feature_np,
        'query_label': query_label,
        'query_cam': query_cam
    }

    scipy.io.savemat('pytorch_result.mat', result)

    print(opt.name)
    result = './model/%s/result.txt' % opt.name
    # os.system('python evaluate.py | tee -a %s' % result)

    if opt.multi:
        multi_query_result = {
            'mquery_f': mquery_feature_np,
            'mquery_label': mquery_label,
            'mquery_cam': mquery_cam
        }
        scipy.io.savemat('multi_query.mat', multi_query_result)
