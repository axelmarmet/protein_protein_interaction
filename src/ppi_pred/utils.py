import os
import numpy as np
import torch
import random
import torch.distributed as dist

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, auc, roc_curve, precision_recall_curve, PrecisionRecallDisplay, average_precision_score, fbeta_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# necessary functions for distributed training
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

#All gather from https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/utils/comm.py making use of picklable data
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        tensor of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    #I modified a bit the end of the code to match our output format
    #data_list = []
    data_tensor = torch.tensor([])
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        #data_list.append(pickle.loads(buffer))
        data_tensor = torch.cat(data_tensor,pickle.loads(buffer))

    #return data_list
    return data_tensor

def plot_confusion_matrix(CM):
    
    """
    Function used in the metrics' function to plot the confusion matrix
    """
    
    TP = CM[1,1]
    TN = CM[0,0]
    FN = CM[1,0]
    FP = CM[0,1]
    
    label = np.asarray([['TP {}'.format(TP), 'FP {}'.format(FP)],
                        ['FN {}'.format(FN), 'TN {}'.format(TN)]])
    
    df_cm = pd.DataFrame([[TP, FP],[FN, TN]], index=['1', '0'], columns=['1_gt', '0_gt']) 
    
    return sns.heatmap(df_cm, cmap='YlOrRd', annot=label, annot_kws={"size": 16}, cbar=False, fmt='')

def metrics(y_pred, y_true, beta = 0.5, threshold = 0.5, curve = False):
    
    """
    Function to compute several metrics
    
    Input:
    -y_pred: tensor of size (batchsize, 1) with the probability prediction of the model
    -y_true: tensor with the target
    -curve: boolean to plot (or not) the ROC curve
    -beta: coefficient for Fbeta-score (The beta parameter determines the weight of recall in the combined score. 
    beta < 1 lends more weight to precision, while beta > 1 favors recall (beta -> 0 considers only precision, beta -> +inf only recall).
    -threshold: thresholding used when computing accuracy, F-score, precision, recall, confusion matrix (prediction 
    values below the threshold are set to class 0, and those above are set to class 1)
    
    Example of input:
    -y_pred: torch.tensor([[0.52],[0.2],[0.85],[0.1],[0.99],[0.78],[0.96]])
    -y_true: torch.tensor([[0],[0],[1],[0],[1],[1],[1]])
    
    Returns:
    -dictionnary: {"acc": acc, "precision": precision, "recall": recall, "fbeta" : fbeta, "CM": CM, "AP_score": AP_score, "AUC_PRC": auc_PRC_score, "AUC_ROC": auc_ROC_score}
    
    Example of output:
    -{'acc': 0.8571428571428571,
      'precision': 0.8,
      'recall': 1.0,
      'fbeta': 0.8333333333333334,
      'CM': array([[2, 1],[0, 4]]),
      'AP_score': 0.8875,
      'AUC_PRC': 0.8708333333333333,
      'AUC_ROC': 0.8333333333333334
      }
    """
    
    ######################################
    ###### NON THRESHOLDED METRICS #######
    ######################################
    
    # ROC CURVE (and AUC)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_ROC_score = auc(fpr, tpr)
    
    # PRECISION-RECALL CURVE (and AUC)
    precision_values, recall_values, _ = precision_recall_curve(y_true, y_pred)
    auc_PRC_score = auc(recall_values, precision_values)
    
    # AVERAGE PRECISION SCORE
    AP_score = average_precision_score(y_true, y_pred)
    
    ######################################
    ######## THRESHOLDED METRICS #########
    ######################################
    
    y_pred_thresholded = (y_pred>threshold).type(torch.uint8)
    
    # CONFUSION MATRIX
    CM = confusion_matrix(y_true, y_pred_thresholded)
    
    # F-BETA SCORE
    fbeta = fbeta_score(y_true, y_pred_thresholded, beta)
    
    # ACCURACY, PRECISION, RECALL
    acc = accuracy_score(y_true, y_pred_thresholded)
    precision = precision_score(y_true, y_pred_thresholded)
    recall = recall_score(y_true, y_pred_thresholded)

    if curve:
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Area = {:.2f}".format(auc_ROC_score))
        plt.show()
        
        disp = PrecisionRecallDisplay(precision=precision_values, recall=recall_values)
        disp.plot(label = "AP: {}".format(AP_score))
        plt.title("Precision-Recall Curve - Area = {:.2f}".format(auc_PRC_score))
        plt.legend()
        plt.show()
        
        plot_confusion_matrix(CM)
    
    return {"acc": acc, "precision": precision, "recall": recall, "fbeta" : fbeta, "CM": CM, "AP_score": AP_score, "AUC_PRC": auc_PRC_score, "AUC_ROC": auc_ROC_score}