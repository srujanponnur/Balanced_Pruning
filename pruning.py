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

class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.5, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def checkdir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero/total)*100,1))


def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0


def prune_by_percentile(percent, resample=False, reinit=False,**kwargs):
        global step
        global mask
        global model

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
                
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        step = 0


def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
# traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform) # this is for balance
traindataset = ImbalanceCIFAR10('./data',train=True,download=True, transform=transform)  # this is for imbalance
# traindataset = ImbalanceCIFAR10
testdataset = datasets.CIFAR10('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=60, shuffle=True, num_workers=0,drop_last=False)
    #train_loader = cycle(train_loader)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=60, shuffle=False, num_workers=0,drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = resnet18().to(device)   

lenet_model = LeNet5().to(device)

model = resnet_model

model.apply(weight_init)

initial_state_dict = copy.deepcopy(model.state_dict()) # check what model points to!

path = './data'
model_string = "lenet"
dataset = 'cifar10'
full_path = f'{path}/{model_string}/{dataset}'

checkdir(f'{full_path}')      # change the directory depending on model

torch.save(model, f"{full_path}/initial_state_dict_lt.pth.tar")

make_mask(model)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

for name, param in model.named_parameters():
    print(name, param.size())

bestacc = 0.0
best_accuracy = 0
ITERATION = 2 # number of cycles of pruning that should be done.
comp = np.zeros(ITERATION,float)
bestacc = np.zeros(ITERATION,float)
step = 0
end_iter = 3 # Number of Epochs
all_loss = np.zeros(end_iter, float)
all_accuracy = np.zeros(end_iter, float)
prune_percent = 10 # 10 percent pruning rate
reinit = False # this is false because we are using lottery ticket
resample = False # resample
lr = 1.2e-3
ITE = 1 # First time running the whole process
valid_freq = 1 # frequency of validation
print_freq = 1 # frequency for printing the loss and accuracy (prints every iteration)


for _ite in range(0, ITERATION):
  if not _ite == 0:
      prune_by_percentile(prune_percent, resample=resample, reinit=reinit)
      original_initialization(mask, initial_state_dict)
      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
  print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")
  comp[_ite] = print_nonzeros(model)
  pbar = tqdm(range(end_iter))
  for iter_ in pbar:
    # Frequency for Testing
    if iter_ % valid_freq == 0:
        accuracy = test(model, test_loader, criterion)
        # Save Weights if accuracy is greater than best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkdir(f"{full_path}") # change the path depending on model
            torch.save(model,f"{full_path}/{_ite}_model_lt.pth.tar")

    # Training
    loss = train(model, train_loader, optimizer, criterion)
    all_loss[iter_] = loss # save loss for that iteration
    all_accuracy[iter_] = accuracy # save accuracy for that iteration
    
    # Frequency for Printing Accuracy and Loss
    if iter_ % print_freq == 0:
        pbar.set_description(
            f'Train Epoch: {iter_}/{end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
#   bestacc[_ite] = best_accuracy
#   print(all_loss, bestacc)
#   plt.plot(np.arange(1,(end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
#   plt.plot(np.arange(1,(end_iter)+1), all_accuracy, c="red", label="Accuracy") 
#   plt.title(f"Loss Vs Accuracy Vs Iterations (CIFAR10,LENET)") 
#   plt.xlabel("Iterations") 
#   plt.ylabel("Loss and Accuracy") 
#   plt.legend() 
#   plt.grid(color="gray") 
#   checkdir(f"{full_path}/plots/lt/lenet/cifar10/")
#   plt.savefig(f"{full_path}/plots/lt/lenet/cifar10/lt_LossVsAccuracy_{_ite}.png", dpi=1200) 
#   plt.close()

  # Storing the plots 
  checkdir(f"{full_path}/dumps/lt/lenet/cifar10/")
  print("coming here in checkdir")
  all_loss.dump(f"{full_path}/dumps/lt/lenet/cifar10/lt_all_loss_{_ite}.dat")
  all_accuracy.dump(f"{full_path}/dumps/lt/lenet/cifar10/lt_all_accuracy_{_ite}.dat")

  # Storing the model mask
  checkdir(f"{full_path}/dumps/lt/lenet/cifar10/")
  with open(f"{full_path}/dumps/lt/lenet/cifar10/lt_mask_{_ite}.pkl", 'wb') as fp:
      pickle.dump(mask, fp)

  best_accuracy = 0   # resetting the variables for next iteration
  all_loss = np.zeros(end_iter,float)
  all_accuracy = np.zeros(end_iter,float)


checkdir(f"{full_path}/dumps/lt/lenet/cifar10/")
comp.dump(f"{full_path}/dumps/lt/lenet/cifar10/lt_compression.dat") # compression numpy array
bestacc.dump(f"{full_path}/dumps/lt/lenet/cifar10/lt_bestaccuracy.dat") # best acc in each iter
print(comp)
# Plotting
a = np.arange(ITERATION)
plt.plot(a, bestacc, c="blue", label="Winning tickets") 
plt.title(f"Test Accuracy vs Unpruned Weights Percentage (lenet,cifar10)") 
plt.xlabel("Unpruned Weights Percentage") 
plt.ylabel("Test accuracy") 
plt.xticks(a, comp, rotation ="vertical") 
plt.ylim(0,100)
plt.legend() 
plt.grid(color="gray") 
checkdir(f"{full_path}/plots/lt/lenet/cifar10/")
plt.savefig(f"{full_path}/plots/lt/lenet/cifar10/lt_AccuracyVsWeights.png", dpi=1200) 
plt.close() 