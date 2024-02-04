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

def train(model, loss, optimizer, x_train, y_train,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size,grad_clip):
    model.to(device)
    x = x_train.clone().detach().to(device)
    y = y_train.clone().detach().to(device)
    model.train()
    optimizer.zero_grad()
    fx = model.forward(x,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size)
    fx = torch.sigmoid(fx).squeeze().to(device)
    y = y.type(torch.FloatTensor).to(device)
    output = loss(fx, y)
    pred_prob=torch.sigmoid(fx)
    output.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
    optimizer.step()
    return output.item(),pred_prob,list(np.array(y_train))


def predict(model, x_val,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size):
    model.to(device)
    model.eval() 
    x_val = x_val.clone().detach().to(device)
    with torch.no_grad():
        output = model(x_val,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size)
    return output


def save_checkpoint(state,is_best,model_path):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, model_path + '/' + 'checkpoint.pth.tar')
    else:
        print("=> Validation Performance did not improve")


def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn,'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred[i])+'\n')

            
if __name__ == '__main__':

    torch.manual_seed(42)
    pos_train_fa = 'data/DNA_MS/txt/4mC/4mC_Tolypocladium/train_pos.txt'
    neg_train_fa = 'data/DNA_MS/txt/4mC/4mC_Tolypocladium/train_neg.txt'
    pos_test_fa = 'data/DNA_MS/txt/4mC/4mC_Tolypocladium/test_pos.txt'
    neg_test_fa = 'data/DNA_MS/txt/4mC/4mC_Tolypocladium/test_neg.txt'

    wordvec_len=4

    X_train, y_train, X_test, y_test = load_train_val_bicoding2(pos_train_fa,neg_train_fa,pos_test_fa,neg_test_fa)
    X_train, y_train, X_test, y_test = load_in_torch_fmt(X_train, y_train, X_test, y_test, wordvec_len)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(X_train.shape)
    print(y_train.shape)

    filter_num='256-256-256-256-256'
    filter_size='9-9-9-9-9'
    pool_size=0
    cnndrop_out=0.5
    if_bn='Y'
    rnn_size=32
    fc_size=0
    max_epochs=50
    lr_decay_step=5
    lr_decay_gamma=0.5
    grad_clip=5
    patience_limit=5
    batch_size=256
    learning_rate=0.01
    
    model_dir_base = filter_num + '_' + filter_size + '_' + str(
                pool_size) + '_' + str(cnndrop_out) \
                        + '_' + if_bn + '_' +\
        str(fc_size) + '_' + str(learning_rate) + '_' + str(batch_size)

    
    n_classes = 2
    n_classes = n_classes - 1
    n_examples = len(X_train)
    
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = torch.nn.BCELoss(reduction='sum')
    
    model = ConvNet_BiLSTM(n_classes,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size,wordvec_len)
    model_dir = 'cnn-rnn/' + model_dir_base +'_'+str(rnn_size)
    model_path = 'Result/' + model_dir
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("> model_dir:",model_path)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    best_acc=0
    patience=0
    
    for i in range(max_epochs):
        start_time = time.time()
        cost = 0.
        y_pred_prob_train = []
        y_batch_train = []
        num_batches = n_examples // batch_size
        
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            output_train,y_pred_prob,y_batch=train(model, loss, optimizer, X_train[start:end], y_train[start:end],filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size,grad_clip)
            cost += output_train
            prob_data = y_pred_prob.cpu().detach().numpy()
            for m in range(len(prob_data)):
                y_pred_prob_train.append(prob_data[m])
            y_batch_train+=y_batch
        start, end=num_batches * batch_size, n_examples
        output_train, y_pred_prob, y_batch = train(model, loss, optimizer, X_train[start:end], y_train[start:end],filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size,grad_clip)
        
        cost += output_train
        prob_data = y_pred_prob.cpu().detach().numpy()
        
        for m in range(len(prob_data)):
            y_pred_prob_train.append(prob_data[m])
        y_batch_train += y_batch
        
        scheduler.step()
        
        fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)

        #plt.plot(fpr_train, tpr_train, label='ROC Curve')
        #plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('ROC Curve')
        #plt.legend()
        #plt.text(0.7, 0.2, 'AUROC = %0.3f' % auc(fpr_train, tpr_train), bbox=dict(facecolor='white', alpha=0.5))
        #plt.show()
        #plt.savefig('Result/ROC_Curve.png')
        
        #Val
        output_val = predict(model, X_val,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size)
        #output_val = predict(model, X_test,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size)
        y_pred_prob_val = []
        y_pred_val=[]
        prob_data=torch.sigmoid(output_val).cpu().detach().numpy()
        
        for m in range(len(prob_data)):
            y_pred_prob_val.append(prob_data[m][0])
            if prob_data[m][0] >= 0.5:
                y_pred_val.append(1)
            else:
                y_pred_val.append(0)

        fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_pred_prob_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        
        #fpr_val, tpr_val, thresholds_val = roc_curve(y_test, y_pred_prob_val)
        #accuracy = accuracy_score(y_test, y_pred_val)

        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print("Epoch %d, cost = %f, AUROC_train = %0.4f, acc_val = %0.4f%%, AUROC_val = %0.4f, "
              % (i + 1, cost / num_batches, auc(fpr_train, tpr_train), 
                 accuracy, auc(fpr_val, tpr_val)))
        print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        
        cur_acc=accuracy
        is_best = bool(cur_acc >= best_acc)
        best_acc = max(cur_acc, best_acc)
        save_checkpoint({
            'epoch': i+1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_acc,
            'optimizer': optimizer.state_dict()
        }, is_best,model_path)

        if not is_best:
            patience+=1
            if patience>=patience_limit:
                break
        else:
            patience=0
    print('> best val_acc:',best_acc)

    #test
    checkpoint = torch.load(model_path + '/' + 'checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
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
    
    print("=>Test:")
    print("acc = %0.4f, AUROC_test = %0.4f, "
            "MCC_test = %0.4f,Specificity_test = %0.4f, "
            "Precision_test = %0.4f,Recall_test = %0.4f, "
            "F1_test = %0.4f"
            % ( accuracy, auc(fpr_test, tpr_test), 
                mcc_test,specificity_test, 
                precision_test,recall_test, 
                f1_test))
    ytest_ypred_to_file(y_test, y_pred_prob_test,
                        model_path + '/' + 'predout_val.tsv')

