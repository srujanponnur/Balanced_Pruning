import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from os import path
import copy

# file_path = "home\\snaray23\\data\\baseline_18_night\\cifar10\\"
file_path = '.\data\jsons_balanced'  # should be balanced 
file_path_aug = '.\data\jsons_aug'   # should be augmented
file_path_imb = '.\data\jsons_imbalanced' # should be imbalanced
num_classes = 10

files = [file_path, file_path_aug, file_path_imb]

def checkdir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

pruning_ratios = [0,10,19,27.1,34.4,40.9,46.8,52.2,56.9,61.2]
model_results = []
classes = [{'precision':[0]*num_classes,'recall':[0]*num_classes,'accuracy':[0]*num_classes,'f1_score':[0]*num_classes} for i in range(num_classes)]
total_accuracy = [0]*num_classes

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
empty = [{'precision':[0]*num_classes,'recall':[0]*num_classes,'accuracy':[0]*num_classes,'f1_score':[0]*num_classes} for i in range(num_classes)]
for model_path in files:
    classes = copy.deepcopy(empty)
    for file in os.listdir(model_path):
        if 'json' in file:
            ite = int(file.split('_')[0])
            with open(f'{model_path}/{file}','r') as f:
                data = json.load(f)
                # print(data)
                for i in range(num_classes):
                    classes[i]['precision'][ite] = data[str(i)]['precision']
                    classes[i]['recall'][ite] = data[str(i)]['recall']
                    classes[i]['f1_score'][ite] = data[str(i)]['f1-score']
                    classes[i]['accuracy'][ite] = data['per_class_accuracy'][i]
                total_accuracy[i] = data.get('accuracy')
    model_results.append(classes)

plot_path = path.join(model_path,'manual_plots')
checkdir(plot_path)




for i in range(num_classes):
    checkdir(f"{plot_path}/{i}")
    plt.plot(pruning_ratios, model_results[0][i]['precision'], c="blue", label="Precision on Balanced Dataset") 
    plt.plot(pruning_ratios, model_results[1][i]['precision'], c="red", label="Precision on Augmented Dataset") 
    plt.plot(pruning_ratios, model_results[2][i]['precision'], c="yellow", label="Precision on Imbalanced Datset") 
    plt.title(f"class-{i} Precision vs Pruning Ratio")
    plt.xlabel("Pruned Ratios") 
    plt.ylabel("Precision") 
    # plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,1)
    plt.legend() 
    plt.grid(color="gray") 
    plt.savefig(f"{plot_path}/{i}/precision_ratio.png", dpi=1200)
    plt.close()

    plt.plot(pruning_ratios, model_results[0][i]['recall'], c="blue", label="Recall on Balanced Dataset") 
    plt.plot(pruning_ratios, model_results[1][i]['recall'], c="red", label="Recall on Augmented Dataset") 
    plt.plot(pruning_ratios, model_results[2][i]['recall'], c="yellow", label="Recall on Imbalanced Dataset") 
    plt.title(f"class-{i} Recall vs Pruning Ratio")
    plt.xlabel("Pruned Ratios") 
    plt.ylabel("Recall") 
    # plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,1)
    plt.legend() 
    plt.grid(color="gray") 
    plt.savefig(f"{plot_path}/{i}/recall_ratio.png", dpi=1200)
    plt.close() 

    plt.plot(pruning_ratios, model_results[0][i]['accuracy'], c="blue", label="Accuracy on Balanced Dataset")
    plt.plot(pruning_ratios, model_results[1][i]['accuracy'], c="red", label="Accuracy on Augmented Dataset")
    plt.plot(pruning_ratios, model_results[2][i]['accuracy'], c="yellow", label="Accuracy on Imbalanced Dataset")
    plt.title(f"class-{i} Accuracy vs Pruning Ratio")
    plt.xlabel("Pruned Ratios")
    plt.ylabel("Accuracy")
    # plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,1)
    plt.legend() 
    plt.grid(color="gray") 
    plt.savefig(f"{plot_path}/{i}/accuracy_ratio.png", dpi=1200)
    plt.close()

    plt.plot(pruning_ratios, model_results[0][i]['f1_score'], c="blue", label="F1-Score on Balanced Dataset")
    plt.plot(pruning_ratios, model_results[1][i]['f1_score'], c="red", label="F1-Score on Augmented Dataset")
    plt.plot(pruning_ratios, model_results[2][i]['f1_score'], c="yellow", label="F1-Score on Imbalanced Dataset")
    plt.title(f"class-{i} F1-Score vs Pruning Ratio")
    plt.xlabel("Pruned Ratios")
    plt.ylabel("F1-Score")
    # plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,1)
    plt.legend()
    plt.grid(color="gray")
    plt.savefig(f"{plot_path}/{i}/f1_score_ratio.png", dpi=1200)
    plt.close()


