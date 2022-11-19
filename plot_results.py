import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from os import path

file_path = "C:\\Users\\sponnur\\Desktop\\codebase\\Balanced_Pruning\\data\\resnet_balance_18th_afternoon\\cifar10\\"
num_classes = 10

def checkdir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

pruning_ratios = [0,10,19,27.1,34.4,40.9,46.8,52.2,56.9,61.2]

classes = [{'precision':[],'recall':[],'accuracy':[],'f1_score':[]} for i in range(num_classes)]
total_accuracy = []

'''
{
    'class':{
        'precision':[]
        'recall':[]
        'recall':[]
        'f1_score':[]
    }
}
'''

pruning_map = {}
print(file_path)
for file in os.listdir(file_path):
    if 'json' in file:
        # ite = int(file.split('_')[0])
        print(file)
        with open(f'{file_path}\{file}','r') as f:
            data = json.load(f)
            print(data)
            for i in range(num_classes):
                classes[i]['precision'].append(data[str(i)]['precision'])
                classes[i]['recall'].append(data[str(i)]['recall'])
                classes[i]['f1_score'].append(data[str(i)]['f1-score'])
                classes[i]['accuracy'].append(data['per_class_accuracy'][i])
            total_accuracy.append(data.get('accuracy'))

plot_path = path.join(file_path,'manual_plots')
checkdir(plot_path)

for i in range(num_classes):
    checkdir(f"{plot_path}/{i}")
    plt.plot(pruning_ratios, classes[i]['precision'], c="blue", label="Precision Plot") 
    plt.title(f"class-{i} Precision vs Pruning Ratio")
    plt.xlabel("Pruned Ratios") 
    plt.ylabel("Precision") 
    # plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,1)
    plt.legend() 
    plt.grid(color="gray") 
    plt.savefig(f"{plot_path}/{i}/precision_ratio.png", dpi=1200)
    plt.close() 

    plt.plot(pruning_ratios, classes[i]['recall'], c="blue", label="Recall Plot") 
    plt.title(f"class-{i} Recall vs Pruning Ratio")
    plt.xlabel("Pruned Ratios") 
    plt.ylabel("Recall") 
    # plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,1)
    plt.legend() 
    plt.grid(color="gray") 
    plt.savefig(f"{plot_path}/{i}/recall_ratio.png", dpi=1200)
    plt.close() 

    plt.plot(pruning_ratios, classes[i]['accuracy'], c="blue", label="Accuracy Plot")
    plt.title(f"class-{i} Accuracy vs Pruning Ratio")
    plt.xlabel("Pruned Ratios")
    plt.ylabel("Accuracy")
    # plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,1)
    plt.legend() 
    plt.grid(color="gray") 
    plt.savefig(f"{plot_path}/{i}/accuracy_ratio.png", dpi=1200)
    plt.close()

    plt.plot(pruning_ratios, classes[i]['f1_score'], c="blue", label="F1-Score tickets")
    plt.title(f"class-{i} F1-Score vs Pruning Ratio")
    plt.xlabel("Pruned Ratios")
    plt.ylabel("F1-Score")
    # plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,1)
    plt.legend()
    plt.grid(color="gray")
    plt.savefig(f"{plot_path}/{i}/precision_ratio.png", dpi=1200)
    plt.close()
    break








