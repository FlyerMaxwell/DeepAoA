#!/Users/caiyunxiang/anaconda3/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from operator import itemgetter

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
#dataset_path = '/Users/caiyunxiang/Desktop/AOA/c25_self'
dataset_path = '/Users/caiyunxiang/Desktop/Self/Self_result/offset_np'
#dataset_path = '/Users/caiyunxiang/Desktop/AOA/6-15-result/offset_np'
#dataset_path = '/Users/caiyunxiang/Desktop/AOA/c2'
#dataset_path = '/Users/caiyunxiang/Desktop/FUCK/c25/union'
model_path = '/Users/caiyunxiang/Desktop/Self/Self_result/Model'
batch_size = 5000
test_batch_size = 2392
hidden_layer = 6
Input_dim, H_1, H_2, H_3, H_4, H_5, H_6, Output_dim = 312, 256, 224, 160, 128, 82, 64, 20
Learning_rate = 1e-3
epoch = 40
cnn = True
cnn2d = False
drop_p = 0.5
print_iter = 50
norm = True

class ArgOffsetDataset(Dataset):
    """CSI argument offset dataset."""

    def __init__(self, dataset_path, datatype='train'):
        self.dataset_path = dataset_path
        self.data_path = os.path.join(self.dataset_path, os.path.join(datatype, 'data.npy'))
        self.labels_path = os.path.join(self.dataset_path, os.path.join(datatype, 'labels.npy'))
        self.data = np.load(self.data_path)
        if norm:
            self.data = self.__z_score__(self.data)
        if cnn:
            if cnn2d:
                self.data = np.reshape(self.data, (self.data.shape[0], 1, 6, 52))
            else:
                self.data = np.reshape(self.data, (self.data.shape[0], 1, self.data.shape[1]))
        self.labels = np.load(self.labels_path)
        print(datatype+"_data shape: ", self.data.shape)
        print(datatype+"_labels shape: ", self.labels.shape)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # return self.train_data[idx], self.train_labels[idx]
        sam = torch.from_numpy(self.data[idx]).type(torch.float32)
        lab = torch.from_numpy(self.labels[idx]).type(torch.long)
        # return self.test_data[idx], self.test_labels[idx]
        return sam, lab
    
    def __norm__(self, dataset):
        print(f'normalization...')
        return dataset/np.std(dataset,axis=0)

    def __norm2__(self, dataset):
        print(f'normalization...')
        return (dataset-np.min(dataset))/(np.max(dataset)-np.min(dataset))

    def __z_score__(self, dataset):
        print(f'z-scoring...')
        dataset -= np.mean(dataset, axis=0)
        dataset /= np.std(dataset, axis=0)
        return dataset

class AoACNN(torch.nn.Module):
    def __init__(self):
        super(AoACNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1), # input [batchsize,1,312], output [batchsize,8,156]
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=3) # input [batchsize,8,156], output [batchsize,8,52]
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1), # input [batchsize,8,52], output [batchsize,16, 26]
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool1d(kernel_size=2, stride=2) # input [batchsize,16,26], output [batchsize,16,13]
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # input [batchsize,16,13], output [batchsize,32, 13]
            torch.nn.BatchNorm1d(32), 
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32*13, 100),
            # torch.nn.Dropout(drop_p),
            torch.nn.ReLU(),
            torch.nn.Linear(100, Output_dim)
        )
        # self.dp = torch.nn.Dropout(drop_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x.view(x.size(0), -1))
        # x = self.dp(x)
        return x

class AoACNN_2d(torch.nn.Module):
    def __init__(self):
        super(AoACNN_2d, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1), # input [batchsize,1,6,52], output [batchsize,8,3,26]
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=3, stride=3) # input [batchsize,8,156], output [batchsize,8,52]
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=(0,1)), # input [batchsize,8,3,26], output [batchsize,16,1,13]
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2) # input [batchsize,16,26], output [batchsize,16,13]
        )
        # self.conv3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # input [batchsize,16,13], output [batchsize,32, 13]
        #     torch.nn.BatchNorm2d(32), 
        #     torch.nn.ReLU(),
        # )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16*13, 100),
            torch.nn.Dropout(drop_p),
            torch.nn.ReLU(),
            torch.nn.Linear(100, Output_dim)
        )
        # self.dp = torch.nn.Dropout(drop_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.fc(x.view(x.size(0), -1))
        # x = self.dp(x)
        return x


def work(mode='Train', cnn=False):
    train_dataset = ArgOffsetDataset(dataset_path, 'train')
    test_dataset = ArgOffsetDataset(dataset_path, 'test')
    trainDataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(test_dataset, batch_size=test_batch_size)
    model_type = ""
    if mode=='Train':
        if cnn:
            if cnn2d:
                model = AoACNN_2d()
                model_type = 'CNN2d'
            else:
                model = AoACNN()
                model_type = 'CNN1d'
        else:
            model = torch.nn.Sequential(
                torch.nn.Linear(Input_dim, H_1),
                torch.nn.ReLU(),
                torch.nn.Linear(H_1, H_2),
                torch.nn.ReLU(),
                torch.nn.Linear(H_2, H_3),
                torch.nn.ReLU(),
                torch.nn.Linear(H_3, H_4),
                torch.nn.ReLU(),
                torch.nn.Linear(H_4, H_5),
                torch.nn.ReLU(),
                torch.nn.Linear(H_5, H_6),
                torch.nn.ReLU(),
                # torch.nn.Linear(H_6, H_7),
                # torch.nn.ReLU(),
                torch.nn.Linear(H_6, Output_dim),
            )
            model_type = 'FCN'
        if norm:
            model_name = f'model_{model_type}_hidden-layer_{hidden_layer}_epoch_{epoch}_lr_{Learning_rate}_norm.pt'
        else:
            model_name = f'model_{model_type}_hidden-layer_{hidden_layer}_epoch_{epoch}_lr_{Learning_rate}.pt'
        loss_fn = torch.nn.CrossEntropyLoss()
        learning_rate = Learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
        # optimizer = torch.optim.Adagrad()
        loss_count = []
        loss_thr = None
        for iter in range(epoch):
            # if loss_thr != None:
            #     if loss_thr <= 0.7 and loss_thr > 0.6 and learning_rate == Learning_rate:
            #         learning_rate /= 5
            #         print('new learing rate: ', learning_rate)
            #         for param_groups in optimizer.param_groups:
            #             param_groups['lr'] = learning_rate
            #     elif loss_thr <= 0.6 and loss_thr > 0.5 and learning_rate == Lsearning_rate/5:
            #         learning_rate /= 10
            #         print('new learing rate: ', learning_rate)
            #         for param_groups in optimizer.param_groups:
            #             param_groups['lr'] = learning_rate
            #     elif loss_thr <= 0.5 and loss_thr > 0.1 and learning_rate == Learning_rate/50:
            #         learning_rate /= 10
            #         print('new learing rate: ', learning_rate)
            #         for param_groups in optimizer.param_groups:
            #             param_groups['lr'] = learning_rate
            #     elif loss_thr <= 0.1:
            #         print('training finished.')
            #         break
            if loss_thr != None:
                if loss_thr <= 0.1:
                    print('training finished.')
                    break

            print('Epoch ', iter, ': ')
            cnt = 0
            loss_sum = 0
            for i, (data_batch, labels_batch) in enumerate(trainDataloader):
                # print(i, data_batch, data_batch.shape, labels_batch, labels_batch.shape)
                # print('Batch ', i, data_batch.shape, labels_batch.shape)
                # print('Batch ', i)
                y_pred = model(data_batch)
                loss = loss_fn(y_pred, labels_batch.reshape(-1))
                if i%print_iter == 0:
                    loss_count.append(loss)
                    print("Iteration: ", i, ", Loss: ", loss.item())
                loss_sum += loss.item()
                # model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                # with torch.no_grad():
                #     for param in model.parameters():
                #         param -= learning_rate * param.grad
                optimizer.step()
                cnt = i
            loss_thr = loss_sum/(cnt+1)

            torch.save(model, os.path.join(model_path, model_name))
            for (test_x, test_y) in testDataloader:
                pred = model(test_x)
                # print(pred)
                acc = np.argmax(pred.detach().numpy(), axis=1) == test_y.detach().numpy().reshape(-1)
                # pred error distribution (degrees)
                err = np.argmax(pred.detach().numpy(), axis=1) - test_y.detach().numpy().reshape(-1)
                # err_index = 
                err_index = np.where(err!=0)
                err_pred = np.argmax(pred.detach().numpy(), axis=1)[err_index]
                err_truelb   = test_y.detach().numpy().reshape(-1)[err_index]
                #lb2deg = list(range(55,71))+[110, 115, 120]+list(range(125, 129))
                #lb2deg = list(range(55,71))+[110, 115, 120]+list(range(125, 131))
                #lb2deg = [i for i in range(65,130,5)]
                lb2deg = list(range(65,100,5))+list(range(105,170,5))

                
                #print('errp: ', [lb2deg[i] for i in err_pred[:30]])
                #print('true: ', [lb2deg[i] for i in err_truelb[:30]])


                # print(acc)
                print("Validate accuracy: ", acc.mean())
                break
        # acc on train dataset
        acc_sum = []
        for i, (train_x, train_y) in enumerate(trainDataloader):
            print('Batch ', i)
            pred = model(train_x)
            # print('pred: ', pred.shape, 'label: ', train_y.shape)
            acc = np.argmax(pred.detach().numpy(), axis=1) == train_y.detach().numpy().reshape(-1)
            # pred_l += list(np.argmax(pred.detach().numpy(), axis=1))
            # print('pred_l: ', pred_l)
            # true_label += list(train_y.detach().numpy().reshape(-1))
            acc_sum.append(acc.mean())
        print('Total train accuracy: ', sum(acc_sum)/len(acc_sum))
        plt.figure('Loss')
        plt.plot(loss_count, label='loss')
        plt.legend()
        plt.show()

    if mode=='Test':
        if cnn:
            if cnn2d:
                model_type = 'CNN2d'
            else:
                model_type = 'CNN1d'
        else:
            model_type = 'FCN'
        if norm:
            model_name = f'model_{model_type}_hidden-layer_{hidden_layer}_epoch_{epoch}_lr_{Learning_rate}_norm.pt'
        else:
            model_name = f'model_{model_type}_hidden-layer_{hidden_layer}_epoch_{epoch}_lr_{Learning_rate}.pt'
        model = torch.load(os.path.join(model_path, model_name))
        model.eval()
        acc_sum = []
        pred_l = []
        true_label = []

        lb2deg = list(range(65,100,5))+list(range(105, 170,5))
        #lb2deg = list(range(55,71))+[110, 115, 120]+list(range(125, 129))
        #lb2deg = list(range(55,71))+[110, 115, 120]+list(range(125, 131))
        #lb2deg = [i for i in range(65,130,5)]
        for i, (test_x, test_y) in enumerate(testDataloader):
            print('Batch ', i)
            pred = model(test_x)
            # print('pred: ', pred.shape, 'label: ', test_y.shape)
            acc = np.argmax(pred.detach().numpy(), axis=1) == test_y.detach().numpy().reshape(-1)
            pred_l += list(np.argmax(pred.detach().numpy(), axis=1))
            # print('pred_l: ', pred_l)
            true_label += list(test_y.detach().numpy().reshape(-1))
            acc_sum.append(acc.mean())
            print("Test accuracy: ", acc.mean())
        print('Total test accuracy: ', sum(acc_sum)/len(acc_sum))
        # plt.figure('Test accuracy')
        # plt.plot(acc_sum, 'o', label='test_accuracy')
        # plt.title('ArgOffset Test Accuracy')
        # plt.legend()
        # plt.show()

        # error analyse
        cnt = len(pred_l)
        print(f'total test samples: {cnt}')
        # print(f'pred: {pred_l}\ntrue: {true_label}')
        pred_l = np.array(pred_l)
        true_label = np.array(true_label)
        for i,_ in enumerate(pred_l):
            pred_l[i] = lb2deg[pred_l[i]]
            true_label[i] = lb2deg[true_label[i]]
        pred_err = np.abs(pred_l-true_label)
        # print(f'err:{list(pred_err)}')
        pred_err_dict = {} # {err_abs: count}
        deg_err_dict = {} # {deg: {err_abs:count}}
        for p in pred_err:
            if p in pred_err_dict.keys():
                pred_err_dict[p] += 1
            else:
                # print(f'error key: {p}')
                pred_err_dict[p] = 1

        for d in lb2deg:
            deg_err_dict[d] = {}
            for i in lb2deg:
                for j in lb2deg:
                    deg_err_dict[d][abs(i-j)] = 0
            for i, l in enumerate(true_label):
                if l == d:
                    deg_err_dict[d][abs(l-pred_l[i])] += 1

        pred_rst = sorted(pred_err_dict.items(), key=itemgetter(0))
        for e, c in pred_rst:
            if c>0:
                print(f'Error: {e}, count: {c}, ratio: {c*100/cnt}%')
        deg_err_rst = {}
        total_all = 0
        for d in lb2deg:
            print(f'Degree: {d}')
            deg_err_rst[d] = sorted(deg_err_dict[d].items(), key=itemgetter(0))
            total = 0
            for _, c in deg_err_rst[d]:
                total += c
            for e, c in deg_err_rst[d]:
                if c > 0:
                    print(f'    Error: {e}, count: {c}, ratio: {c*100/total}%')
            total_all += total
        # print(deg_err_dict)
        print(f'total samples: {total_all}')
        #print(test_y)Degree: 110

if __name__ == "__main__":
    #work(mode='Train', cnn=True)
    work(mode='Test', cnn=True)
