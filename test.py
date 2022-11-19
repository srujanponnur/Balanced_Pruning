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
import cv2
import glob
from torch.utils.data import Dataset
from torchmetrics.classification import MulticlassAccuracy
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('checking if the env is ready!', device)
print(torch.cuda.device_count(),torch.cuda.get_device_name(0))