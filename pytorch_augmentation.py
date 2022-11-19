import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
import json
from sklearn.metrics import classification_report
from torchmetrics.classification import MulticlassAccuracy
import torchvision.transforms as T
from PIL import Image as im
# def imbalanced_data(dataset,data_loader, cls_num,ratio,n_cpu,batch_size):

#     cls_num=cls_num.split('_')
#     idx_candi=np.array([]).astype(int)

#     for num_candi in cls_num:
#         if opt.dataset == 'MNIST':
#             candi=np.where(data_loader.dataset.targets.numpy() == int(num_candi))[0]

#         elif opt.dataset == 'cifar10':
#             candi=np.where(torch.as_tensor(data_loader.dataset.targets).numpy() == int(num_candi))[0]
            
#         idx_candi=np.append(idx_candi,candi)

#     print(idx_candi)

#     num_of_choice=int(len(idx_candi)*ratio)
#     print('num_of_choice :', num_of_choice)
    
#     idx_=np.random.choice(idx_candi,num_of_choice,replace=False)
#     new_dataset=copy.deepcopy(dataset)
    
#     if opt.dataset == 'MNIST':
#         new_dataset.targets = torch.from_numpy(data_loader.dataset.targets.numpy()[idx_])
#         new_dataset.data = torch.from_numpy(data_loader.dataset.data.numpy()[idx_])

#     elif opt.dataset == 'cifar10':
#         new_dataset.targets = torch.as_tensor(data_loader.dataset.targets).numpy()[idx_]
#         new_dataset.data = torch.as_tensor(data_loader.dataset.data).numpy()[idx_]
            
#     new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu,drop_last=True)
    
#     if opt.dataset == 'MNIST':
#         num_list=[len(np.where(new_data_loader.dataset.targets.numpy()==x)[0]) for x in range(10) ]
    
#     elif opt.dataset =='cifar10':
#         num_list=[len(np.where(torch.as_tensor(new_data_loader.dataset.targets).numpy()==x)[0]) for x in range(10) ]
    
#     print('num_of_sample_class :', num_list)
#     print('ratio_of_sample_class :', np.around(np.array(num_list)/len(dataset),2))
#     return new_dataset,new_data_loader

class ImbalanceCIFAR10_1(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='manual', imb_factor=0.5, rand_number=42, train=True,
                 transform=None, target_transform=None, download=False, manual_class=[0, 3, 8]):
        super(ImbalanceCIFAR10_1, self).__init__(root, train, transform, target_transform, download)
        self.manual_class = manual_class
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == 'manual':
            for index in range(cls_num):
              if index in self.manual_class:
                img_num_per_cls.append(3500)
              else:
                img_num_per_cls.append(int(img_max))
            transformations = GetTransforms()
            train_transforms = transforms.Compose(transformations.trainparams())
            # test_transforms = transforms.Compose(transformations.testparams())
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data 
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='manual', imb_factor=0.5, rand_number=42, train=True,
                 transform=None, target_transform=None, download=False, manual_class=[0, 3, 8]):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.manual_class = manual_class
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == 'manual':
            for index in range(cls_num):
              if index in self.manual_class:
                img_num_per_cls.append(3500)
              else:
                img_num_per_cls.append(int(img_max))
            transformations = GetTransforms()
            train_transforms = transforms.Compose(transformations.trainparams())
            # test_transforms = transforms.Compose(transformations.testparams())
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data.append(temp_data)
        new_targets.extend(temp_data)

        new_data = np.vstack(new_data)
        self.data = new_data 
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class GetTransforms():
    '''Returns a list of transformations when type as requested amongst train/test
       Transforms('train') = list of transforms to apply on training data
       Transforms('test') = list of transforms to apply on testing data'''

    def __init__(self):
        pass

    def trainparams(self):
        train_transformations = [ #resises the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            transforms.RandomRotation((-7,7)),     #Rotates the image to a specified angel
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
            # transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)) #Normalize all the images
            ]

        return train_transformations

    def testparams(self):
        test_transforms = [
            transforms.ToTensor(),
            # transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ]
        return test_transforms

transformations = GetTransforms()
train_transforms = transforms.Compose(transformations.trainparams())
test_transforms = transforms.Compose(transformations.testparams())


class MyData(torchvision.datasets.CIFAR10):
    def __init__(self, data, target, transform=transforms.Compose(transformations.trainparams())):
        self.data = data
        self.target = target
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if (y == 0) and self.transform: # check for minority class
            x = self.transform(x)
            print(x.shape)
        return x, y

transform_general = transforms.Compose([transforms.ToTensor()])
train = torchvision.datasets.CIFAR10('./data/imbalance',train=True, download=True, transform=transform_general) 
# train_2 = MyData(train.data, train.targets)
train_loader = torch.utils.data.DataLoader(train, batch_size=60, shuffle=True, num_workers=0,drop_last=False)

data = torch.as_tensor(train_loader.dataset.data)
targets = torch.as_tensor(train_loader.dataset.targets)
temp_data = data[0:5]
temp_target = targets[0:5]
flipped = []
# iter = 0
data_arr = np.array(data, dtype=np.uint8)
# for i in range(49000):
#     if targets[i] == 8:
#         iter = iter + 1
#         fake_save = im.fromarray(data_arr[i], 'RGB')
#         fake_save.save(f"/home/snaray23/591/orig_ships/orig_ships{iter}.jpeg")

#         if iter == 1500:
#             break
import random
iter = 500
for i in range(250):
    iter = iter + 1
    # jitter = T.ColorJitter(brightness= random.uniform(0,0.5), hue=random.uniform(0,0.5))
    # jitter = T.ColorJitter(brightness= random.uniform(0,0.5), hue=random.uniform(0,0.5))
    # jitter = T.RandomPerspective(distortion_scale=random.uniform(0.5,0.9), p=1.0)
    # jitter = T.RandomAffine(degrees=random.uniform(30,70), translate=(random.uniform(0.1,0.5),random.uniform(0.1,0.5)), scale=(0.5, 0.75))
    # jitter = T.ElasticTransform(alpha=100.0)
    jitter = T.RandomSolarize(threshold=random.uniform(150.0,180.0))
    # jitter = T.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 5))
    pil_img = im.open(f'/home/snaray23/591/orig_ships/orig_ships{iter}.jpeg')
    flipped = None
    #with open(f'/home/snaray23/591/orig_airplanes/orig_planes{iter}.jpeg') as f:
    #    flipped = jitter(f)
    flipped = jitter(pil_img)
    #tempp = im.fromarray(flipped, 'RGB')
    flipped.save(f"/home/snaray23/591/aug_ship/aug_ship{iter}.jpeg")
    plt.imshow(flipped)
    # plt.show()

# train = ImbalanceCIFAR10('./data/imbalance',train=True, download=True, transform=transform_general)
print(len(train))


