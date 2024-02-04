import torch
from matplotlib import pyplot as plt
from torch import optim
import numpy as np
import argparse
from sklearn.metrics import roc_curve,confusion_matrix, matthews_corrcoef,precision_score, recall_score, f1_score,accuracy_score
from sklearn.metrics import auc
import os
import torch.nn.functional as F
import time
import sys
from data_process import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos_train_fa = 'data/DNA_MS/txt/6mA/6mA_A.thaliana/train_pos.txt'
neg_train_fa = 'data/DNA_MS/txt/6mA/6mA_A.thaliana/train_neg.txt'
pos_test_fa = 'data/DNA_MS/txt/4mC/4mC_C.equisetifolia/test_pos.txt'
neg_test_fa = 'data/DNA_MS/txt/4mC/4mC_C.equisetifolia/test_neg.txt'
wordvec_len=4

X_train, y_train, X_test, y_test = load_train_val_bicoding2(pos_train_fa,neg_train_fa,pos_test_fa,neg_test_fa)
X_train, y_train, X_test, y_test = load_in_torch_fmt(X_train, y_train, X_test, y_test, wordvec_len)

filter_num='256-256-256-256-256'
filter_size='9-9-9-9-9'
pool_size=0
cnndrop_out=0.5
if_bn='Y'
rnn_size=32
fc_size=0
batch_size=256
learning_rate=0.01

model_dir_base = filter_num + '_' + filter_size + '_' + str(pool_size) + '_' + str(cnndrop_out) + '_' + if_bn + '_' +str(fc_size) + '_' + str(learning_rate) + '_' + str(batch_size)

n_classes = 2
n_classes = n_classes - 1

model = ConvNet_BiLSTM(n_classes,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size,wordvec_len)
model_dir = 'cnn-rnn/' + model_dir_base +'_'+str(rnn_size)
model_path = 'Result/' + model_dir
checkpoint = torch.load(model_path + '/' + 'checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)
X_test = X_test.to(device)
with torch.no_grad():
    output_test = model(X_test,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size)
    
y_pred_prob_test = []
y_pred_test=[]
prob_data=torch.sigmoid(output_test).cpu().detach().numpy()
for m in range(len(prob_data)):
    y_pred_prob_test.append(prob_data[m][0])
    if prob_data[m][0] >= 0.5:
        y_pred_test.append(1)
    else:
        y_pred_test.append(0)  
    
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
mcc_test = matthews_corrcoef(y_test, y_pred_test)
specificity_test = conf_matrix_test[0, 0] / (conf_matrix_test[0, 0] + conf_matrix_test[0, 1])
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
accuracy = accuracy_score(y_test, y_pred_test)
    
print("=>Predict:")
print("acc = %0.4f, AUROC_test = %0.4f, "
        "MCC_test = %0.4f,Specificity_test = %0.4f, "
        "Precision_test = %0.4f,Recall_test = %0.4f, "
        "F1_test = %0.4f"
        % ( accuracy, auc(fpr_test, tpr_test), 
            mcc_test,specificity_test, 
            precision_test,recall_test, 
            f1_test))
