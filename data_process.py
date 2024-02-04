from Bio import SeqIO
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def convert_seq_to_bicoding(seq):
    feat_bicoding=[]
    bicoding_dict={'AC':[1,0,1,4],'GC':[-1,0,2,4],'TC':[0,1,3,4],'CC':[0,-1,4,4],
                   'AT':[1,0,1,3],'GT':[-1,0,2,3],'TT':[0,1,3,3],'CT':[0,-1,4,3],
                   'AG':[1,0,1,2],'GG':[-1,0,2,2],'TG':[0,1,3,2],'CG':[0,-1,4,2],
                   'AA':[1,0,1,1],'GA':[-1,0,2,1],'TA':[0,1,3,1],'CA':[0,-1,4,1]}
    if len(seq)<41:
        seq=seq+'N'*(41-len(seq)) #如果序列长度不足，用N进行填充
    for i in range(0, len(seq) - 1, 1):
        two_chars = seq[i:i+2]
        #print(two_chars)
        if two_chars in bicoding_dict:
            feat_bicoding+=bicoding_dict[two_chars]
        else:
            feat_bicoding+=[0,0,0,0]
    #print(feat_bicoding)
    return feat_bicoding

def convert_seq_to_MNBE(seq):
    feat_bicoding=[]
    bicoding_dict={'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    if len(seq)<41:
        seq=seq+'N'*(41-len(seq))
    for each_nt in seq:
        feat_bicoding+=bicoding_dict[each_nt] 
    return feat_bicoding

def convert_seq_to_NCPNF(seq):
    feat_bicoding=[]
    n=0
    a=0
    c=0
    g=0
    t=0
    bicoding_dict={'A':[1,1,1],'C':[0,0,1],'G':[1,0,0],'T':[0,1,0]}
    if len(seq)<41:
        seq=seq+'N'*(41-len(seq)) #如果序列长度不足，用N进行填充
    for each_nt in seq:
        n=n+1
        feat_bicoding+=bicoding_dict[each_nt]
        if(each_nt=='A'):
            a=a+1
            feat_bicoding.append(a / n)
        if(each_nt=='C'):
            c=c+1
            feat_bicoding.append(c / n)
        if(each_nt=='G'):
            g=g+1
            feat_bicoding.append(g / n)
        if(each_nt=='T'):
            t=t+1
            feat_bicoding.append(t / n)
    return feat_bicoding

def load_data_fasta(in_fa):
    data=[]
    for record in SeqIO.parse(in_fa, "fasta"):
        seq=str(record.seq)
        bicoding = convert_seq_to_bicoding(seq)
        #bicoding = convert_seq_to_MNBE(seq)
        #bicoding = convert_seq_to_NCPNF(seq)
        data.append(bicoding)
    return data

def load_data_txt(in_txt):
    data=[]
    file_path = in_txt
    with open(file_path, 'r') as file:
        for line in file:
            line = str(line.strip())  
            if line:  
                bicoding = convert_seq_to_bicoding(line)
                #bicoding = convert_seq_to_MNBE(line)
                #bicoding = convert_seq_to_NCPNF(line)
                data.append(bicoding)
    return data

def load_train_val_bicoding(pos_train_fa,neg_train_fa):
    data_pos_train = []
    data_neg_train = []
    data_pos_train = load_data_txt(pos_train_fa)
    data_neg_train = load_data_txt(neg_train_fa)
    data_train = np.array([_ + [1] for _ in data_pos_train] + [_ + [0] for _ in data_neg_train])
    np.random.seed(42)
    np.random.shuffle(data_train)
    X = np.array([_[:-1] for _ in data_train])
    y = np.array([_[-1] for _ in data_train])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 / 8, random_state=42)
    return X_train,y_train,X_test,y_test

def load_train_val_bicoding2(pos_train_fa,neg_train_fa,pos_test_fa,neg_test_fa):
    data_pos_train = []
    data_neg_train = []
    data_pos_test = []
    data_neg_test = []
    data_pos_train = load_data_txt(pos_train_fa)
    data_neg_train = load_data_txt(neg_train_fa)
    data_pos_test = load_data_txt(pos_test_fa)
    data_neg_test = load_data_txt(neg_test_fa)

    data_train = np.array([_ + [1] for _ in data_pos_train] + [_ + [0] for _ in data_neg_train])
    data_test = np.array([_ + [1] for _ in data_pos_test] + [_ + [0] for _ in data_neg_test])
    np.random.seed(42)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)
    X_train = np.array([_[:-1] for _ in data_train])
    y_train = np.array([_[-1] for _ in data_train])
    X_test = np.array([_[:-1] for _ in data_test])
    y_test = np.array([_[-1] for _ in data_test])
    return X_train,y_train,X_test,y_test



def load_in_torch_fmt(X_train, y_train, X_test, y_test,vec_len):
    X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]/vec_len), vec_len)
    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]/vec_len), vec_len)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    X_test = torch.from_numpy(X_test).float()
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    print('test')
