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
from model2 import *
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import umap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
pos_train_fa = 'data/DNA_MS/txt/4mC/4mC_C.equisetifolia/train_pos.txt'
neg_train_fa = 'data/DNA_MS/txt/4mC/4mC_C.equisetifolia/train_neg.txt'
pos_test_fa = 'data/DNA_MS/txt/4mC/4mC_C.equisetifolia/test_pos.txt'
neg_test_fa = 'data/DNA_MS/txt/4mC/4mC_C.equisetifolia/test_neg.txt'
wordvec_len=4

#pos_train_x,pos_train_y, neg_train_x, neg_train_y = load_train_val_bicoding3(pos_train_fa,neg_train_fa,wordvec_len)

X_train, y_train, X_test, y_test = load_train_val_bicoding2(pos_train_fa,neg_train_fa,pos_test_fa,neg_test_fa)
X_train, y_train, X_test, y_test = load_in_torch_fmt(X_train, y_train, X_test, y_test, wordvec_len)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset,batch_size=256,shuffle=True)

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
batch_size=32
learning_rate=0.01
    
model_dir_base = filter_num + '_' + filter_size + '_' + str(
            pool_size) + '_' + str(cnndrop_out) \
                    + '_' + if_bn + '_' +\
    str(fc_size) + '_' + str(learning_rate) + '_' + str(batch_size)

    
n_classes = 2

loss = torch.nn.CrossEntropyLoss(reduction='sum')
n_classes = n_classes - 1
loss = torch.nn.BCELoss(reduction='sum')
    
model = ConvNet_BiLSTM(n_classes,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size,wordvec_len)
model_dir = 'cnn-rnn/' + model_dir_base +'_'+str(rnn_size)
model_path = 'Result/' + model_dir
    
if not os.path.exists(model_path):
    os.makedirs(model_path)
    print("> model_dir:",model_path)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
batch_size = batch_size
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

checkpoint = torch.load(model_path + '/' + 'checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

model.to(device)
model.eval()  # 切换到评估模式

# 收集特征向量的代码逻辑
feature_vectors = []
all_labels = []  # 改动之处：初始化用于收集所有标签的列表

with torch.no_grad():
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        feature_vector = model(inputs, filter_size, filter_num, pool_size, if_bn, cnndrop_out, rnn_size, fc_size)
        
        # 将特征向量从 GPU 转移到 CPU 并转换为 NumPy 数组
        feature_vector = feature_vector.cpu().numpy()
        feature_vectors.append(feature_vector)
        all_labels.append(labels.cpu().numpy())  # 改动之处：收集当前批次的标签

feature_vectors = np.concatenate(feature_vectors, axis=0)
all_labels = np.concatenate(all_labels, axis=0)  # 改动之处：将所有批次的标签合并

def tSNE1(vectors,labels):
    # t-SNE 需要花费一些时间来运行，尤其是对于较大的数据集
    # 适当地设置 `n_iter` 和 `perplexity` 可以改善结果和运行时间
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(feature_vectors)
    # 创建一个绘图
    plt.figure(figsize=(10, 8))
    # 使用 seaborn 的方便性创建散点图
    # all_labels 是标签列表：0 代表负样本，1 代表正样本
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=all_labels,
        palette=sns.color_palette("hls", 2),
        legend="full",
    )
    plt.legend(title='Sample Type', loc='best', labels=['Negative Samples', 'Positive Samples'])
    plt.savefig('Result/4mC_C.equisetifolia_tSNE.png')

def UMAP1(vectors,labels):
    # 使用UMAP进行降维
    umap_embeddings = umap.UMAP().fit_transform(feature_vectors)

    # 绘制散点图
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=all_labels)
    plt.colorbar()
    plt.savefig('Result/4mC_C.equisetifolia_tSNE.png')

def PCA1(vectors,labels):
    # 假设 feature_vectors 的维度为 (n_samples, n_features, n_dimensions)
    # 首先将其 reshape 成 (n_samples, n_features * n_dimensions)
    #reshaped_feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)
    pca = PCA(n_components=2)
    pca_representation = pca.fit_transform(feature_vectors)

    # 改动之处：这里 all_labels 而不是 labels
    plt.scatter(pca_representation[all_labels == 1, 0], pca_representation[all_labels == 1, 1], label='Positive Samples',s=10)
    plt.scatter(pca_representation[all_labels == 0, 0], pca_representation[all_labels == 0, 1], label='Negative Samples',s=10)

    plt.legend()
    plt.savefig('Result/4mC_C.equisetifolia_pca.png')
#tSNE1(feature_vectors,all_labels)
#PCA1(feature_vectors,all_labels)
UMAP1(feature_vectors,all_labels)