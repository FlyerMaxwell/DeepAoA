#!/Users/caiyunxiang/anaconda3/bin/python
# -*- coding: utf-8 -*-

# accept processed (n, 52) data from data_process.py, split for train and test, and do some other processing.

import os
import json
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from aoa_utils import *    # import global variables

class Data():
    def __init__(self):
        self.info = self.load_info()
        self.train_path = os.path.join(dataset_path, 'train')
        self.test_path = os.path.join(dataset_path, 'test')
        self.train_sep_path = os.path.join(dataset_sep_path, 'train')
        self.test_sep_path = os.path.join(dataset_sep_path, 'test')
        self.data = {}
        self.deg_data = {}
        self.ant_data = None
        self.dataset = {}
        self.train_ratio = 0.8
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.sep_dataset = {}
        
#把已有的数据集load进去
    def load_dataset(self):
        train_file_path = os.path.join(self.train_path, 'data.npy')
        train_label_path = os.path.join(self.train_path, 'labels.npy')
        test_file_path = os.path.join(self.test_path, 'data.npy')
        test_label_path = os.path.join(self.test_path, 'labels.npy')
        print('loading shuffled data and labels...')
        self.train_data = np.load(train_file_path)
        self.train_labels = np.load(train_label_path)
        self.test_data = np.load(test_file_path)
        self.test_labels = np.load(test_label_path)
        print('shuffled data and labels loaded.')
        print('train_data shape: ', self.train_data.shape)
        print('train_labels shape: ', self.train_labels.shape)
        print('test_data shape: ', self.test_data.shape)
        print('test_labels shape: ', self.test_labels.shape)

    def load_dataset_each(self):
        for deg in self.info['degs']:
            self.sep_dataset[deg] = {}
            for dis in list(self.info[deg]['10.2'].keys()):#这里怎么写死成10.2了？？
                sep_path = os.path.join(dataset_sep_path, deg,'_', dis, '.npy')
                print('loading separated data: ',sep_path)
                self.sep_dataset[deg][dis] = np.load(sep_path)
                print(self.sep_dataset[deg][dis].shape)
    
    def load_info(self):
        info_file = os.path.join(info_path, 'info.json')
        print('Loading data_info...')
        with open(info_file, 'r') as fp:
            info = json.load(fp)
        print("Data_info loaded.")
        # to load npy, file path is os.path.join(data[deg][ant][dis]['arg'], '.npy')
        return info

    def load_arg_data(self, info, avg):
        # merge data from different distances for each degree
        for deg in info['degs']:
            #if deg=='deg63':
                #continue
            for ant in list(info[deg].keys()):
                if ant not in ants:
                    continue
                for dis in list(info[deg][ant].keys()):
                    #print("info[deg][ant][dis]['pkt_range'] is",info[deg][ant][dis]['pkt_range'])
                    if avg:
                        store_file = info[deg][ant][dis]['avg_arg']+'.npy'
                    else:
                        store_file = info[deg][ant][dis]['arg']+'.npy'
                    file_data = np.load(store_file)
                    if type(self.ant_data) == type(None):
                        self.ant_data = file_data
                    else:
                        self.ant_data = np.vstack((self.ant_data, file_data))
                    #print("store_file is :", store_file, "file_data.shape is :",file_data.shape)
                # if ant in self.deg_data.keys():
                #     self.deg_data[ant] = np.vstack((self.deg_data[ant], self.ant_data))
                # else:
                self.deg_data[ant] = self.ant_data
                self.ant_data = None
            self.data[deg] = self.deg_data
            self.deg_data = {}  # Do NOT use clear()! Or self.data[deg] will also be erased because that it's a reference rather than a copy to self.deg_data
        for deg in info['degs']:
            #if deg=='deg63':
               # continue
            for ant in list(info[deg].keys()):
                if ant not in ants:
                    continue
                print(self.data[deg][ant].shape)#索引方式

    def load_arg_data_each(self, info, avg):
        # do not merge data from different distance for each degree, keep them seperated
        for deg in info['degs']:
            for ant in list(info[deg].keys()):
                if ant not in ants:
                    continue
                for dis in list(info[deg][ant].keys()):
                    print(info[deg][ant][dis]['pkt_range'])
                    if avg:
                        store_file = info[deg][ant][dis]['avg_arg']+'.npy'
                    else:
                        store_file = info[deg][ant][dis]['arg']+'.npy'
                    file_data = np.load(store_file)
                    if type(self.ant_data) == type(None):
                        self.ant_data = {dis: file_data}
                    else:
                        self.ant_data[dis] = file_data
                    print(store_file, file_data.shape)
                # if ant in self.deg_data.keys():
                #     self.deg_data[ant] = np.vstack((self.deg_data[ant], self.ant_data))
                # else:
                self.deg_data[ant] = self.ant_data
                self.ant_data = None
            self.data[deg] = self.deg_data
            self.deg_data = {}  # Do NOT use clear()! Or self.data[deg] will also be erased because that it's a reference rather than a copy to self.deg_data
        for deg in info['degs']:
            for ant in list(info[deg].keys()):
                if ant not in ants:
                    continue
                for dis in info[deg][ant].keys():
                    print(self.data[deg][ant][dis].shape)

    def gen_dataset(self):
        """Generate offset dataset."""
        for deg in self.info['degs']:
            #if deg=='deg63':
              #  continue
            self.dataset[deg] = None
            for i, anti in enumerate(ants):
                for antj in ants[i+1:]:
                    #print(anti, self.data[deg][anti].shape, antj, self.data[deg][antj].shape)
                    offset = self.data[deg][antj] - self.data[deg][anti]
                    if type(self.dataset[deg]) == type(None):
                        self.dataset[deg] = offset
                    else:
                        self.dataset[deg] = np.hstack((self.dataset[deg], offset))
            #print(deg, self.dataset[deg].shape)
            print("saving dataset for ",deg, "...")
            np.save(os.path.join(dataset_path, deg), self.dataset[deg])
            print('dataset saved.')

    def gen_dataset_each(self):
        """Generate offset dataset for each distance."""
        for deg in self.info['degs']:
            self.sep_dataset[deg] = {}
            for i, anti in enumerate(list(self.info[deg].keys())):
                if anti not in ants:
                    continue
                for antj in ants[i+1:]:
                    for dis in self.data[deg][anti].keys():
                        #print(deg, dis, anti, self.data[deg][anti][dis].shape, antj, self.data[deg][antj][dis].shape)
                        offset = self.data[deg][antj][dis] - self.data[deg][anti][dis]
                        if not dis in self.sep_dataset[deg].keys():
                            self.sep_dataset[deg][dis] = offset
                        else:
                            self.sep_dataset[deg][dis] = np.hstack((self.sep_dataset[deg][dis], offset))
            for dis in self.sep_dataset[deg].keys():
                #print(deg, dis, self.sep_dataset[deg][dis].shape)
                print("saving dataset for ",deg, dis, "...")
                np.save(os.path.join(dataset_sep_path, deg+'_'+dis), self.sep_dataset[deg][dis])
                print('dataset saved.')
    
    def plot_mean(self, sep=True, y_min=-1, y_max=1):
        if sep:
            dataset = self.sep_dataset
        else:
            dataset = self.dataset
        for deg in dataset.keys():
            # plt.ion()
            for dis in dataset[deg].keys():
                print(deg, dis, dataset[deg][dis].shape)
                print("plotting mean for ",deg, dis, "...")
                tmp = np.mean(dataset[deg][dis], axis=0)
                tmp = np.reshape(tmp,(6, 52))
                fig = plt.figure('Argument offset mean, deg: %s, dis: %s'%(deg, dis), figsize=(200,6))
                sub = fig.add_subplot(1,1,1)
                sub.set_ylim(y_min, y_max)
                title = ['ant2-ant1', 'ant3-ant1', 'ant4-ant1', 'ant3-ant2', 'ant4-ant2', 'ant4-ant3']
                for j in range(6):
                    sub.plot(tmp[j], '-*', label=title[j])
                    sub.legend()
                plt.show()
                # plt.pause(5)
                # plt.close()

    def plot_packet_mean(self, sep=True, y_min=-1, y_max=1):
        if sep:
            dataset = self.sep_dataset
        else:
            dataset = self.dataset
        for deg in dataset.keys():
            plt.ion()
            for dis in dataset[deg].keys():
                if dis != 'd4.35':
                    continue
                #print(deg, dis, dataset[deg][dis].shape)
                for i in range(dataset[deg][dis].shape[0]//68):
                    print("plotting mean for deg: ", deg, "dis: ", dis, "pkt_no: ", i, "...")
                    tmp = np.mean(dataset[deg][dis][i*68:i*68+68], axis=0)
                    tmp = np.reshape(tmp,(6, 52))
                    fig = plt.figure('Argument offset mean, deg: {deg}, dis: {dis}, pkt_no: {i}', figsize=(200,6))
                    sub = fig.add_subplot(1,1,1)
                    sub.set_ylim(y_min, y_max)
                    title = ['ant2-ant1', 'ant3-ant1', 'ant4-ant1', 'ant3-ant2', 'ant4-ant2', 'ant4-ant3']
                    for j in range(6):
                        sub.plot(tmp[j], '-*', label=title[j])
                        sub.legend()
                    # plt.show()
                    plt.pause(0.05)
                    plt.close()

    # deprecated
    # def denoise(self):
    #     # Should this step be here? Or in data_process.py?
    #     if len(self.dataset) == 0:
    #         print("loading dataset...")
    #         for deg in self.info['degs']:
    #             self.dataset[deg] = np.load(os.path.join(dataset_path, deg+'.npy'))
    #         print("dataset loaded.")
    #     print('Denoising dataset...')
    #     # for deg in self.info['degs']:
    #     #     # do real work
    #     #     self.dataset[deg]
    #     print("test for deg35.")
    #     # dyn_plot(self.dataset, 'deg35', freq=50, beg=400, end=600)
    #     sta_plot(self.dataset, 'deg35', custom=[407,408,409])#[443,444,445,446])



    def split_dataset(self, save=False):
        if len(self.dataset) == 0:
            print("loading dataset...")
            for deg in self.info['degs']:
                self.dataset[deg] = np.load(os.path.join(dataset_path, deg+'.npy'))
            print("dataset loaded.")
        print('splitting dataset: training ratio: ', self.train_ratio, ', test ratio: ', 1-self.train_ratio)
        print('Generating training set...')
        labeled_data = None
        labeled_labels = None
        print("self.dataset.keys()",self.dataset.keys())
        for i,deg in enumerate(self.dataset.keys()):
            print(self.dataset[deg].shape)
            labels = np.zeros(self.dataset[deg].shape[0])+i
            labels = labels.reshape(-1,1)
            #print(labels.shape, labels)
            if type(labeled_data) == type(None):
                labeled_data = self.dataset[deg]
                labeled_labels = labels
            else:
                labeled_data = np.vstack((labeled_data, self.dataset[deg]))
                labeled_labels = np.vstack((labeled_labels, labels))
        print("data size: ", labeled_data.shape, "label size: ", labeled_labels.shape)
        perm = np.random.permutation(np.array(range(0,labeled_data.shape[0])))
        print(perm, perm.shape)
        shuffled_data = labeled_data[perm]
        shuffled_labels = labeled_labels[perm]
        train_size = int(labeled_data.shape[0]*self.train_ratio)
        test_size = labeled_data.shape[0] - train_size
        if(save):
            train_file_path = os.path.join(self.train_path, 'data')
            train_label_path = os.path.join(self.train_path, 'labels')
            test_file_path = os.path.join(self.test_path, 'data')
            test_label_path = os.path.join(self.test_path, 'labels')
            print('saving shuffled data and labels...')
            np.save(train_file_path, shuffled_data[:train_size])
            np.save(train_label_path, shuffled_labels[:train_size])
            np.save(test_file_path, shuffled_data[train_size:])
            np.save(test_label_path, shuffled_labels[train_size:])
            print('shuffled data and labels saved.')

    def split_dataset_each(self, save=False, dis='d4.35'):
        if len(self.sep_dataset) == 0:
            print("loading dataset...")
            self.load_dataset_each()
            print("dataset loaded.")
        print('splitting dataset: training ratio: ', self.train_ratio, ', test ratio: ', 1-self.train_ratio)
        print('Generating training set...')
        labeled_data = None
        labeled_labels = None
        dataset = self.sep_dataset
        for i,deg in enumerate(dataset.keys()):
            print(dataset[deg][dis].shape)
            labels = np.zeros(dataset[deg][dis].shape[0])+i
            labels = labels.reshape(-1,1)
            print(labels.shape, labels)
            if type(labeled_data) == type(None):
                labeled_data = dataset[deg][dis]
                labeled_labels = labels
            else:
                labeled_data = np.vstack((labeled_data, dataset[deg][dis]))
                labeled_labels = np.vstack((labeled_labels, labels))
        print("data size: ", labeled_data.shape, "label size: ", labeled_labels.shape)
        perm = np.random.permutation(np.array(range(0,labeled_data.shape[0])))
        print(perm, perm.shape)
        shuffled_data = labeled_data[perm]
        shuffled_labels = labeled_labels[perm]
        train_size = int(labeled_data.shape[0]*self.train_ratio)
        test_size = labeled_data.shape[0] - train_size
        if(save):
            train_file_path = os.path.join(self.train_sep_path, 'data')
            train_label_path = os.path.join(self.train_sep_path, 'labels')
            test_file_path = os.path.join(self.test_sep_path, 'data')
            test_label_path = os.path.join(self.test_sep_path, 'labels')
            print('saving shuffled data and labels...')
            np.save(train_file_path, shuffled_data[:train_size])
            np.save(train_label_path, shuffled_labels[:train_size])
            np.save(test_file_path, shuffled_data[train_size:])
            np.save(test_label_path, shuffled_labels[train_size:])
            print('shuffled data and labels saved.')

def dyn_plot(dataset, deg=None, freq=50, beg=None, end=None, custom=None):
    """freq: update frequency, default: 50 Hz"""
    if deg==None:
        print('Please choose the deg you want to plot!')
        return

    if beg != None and end != None:
        idx_range = range(beg, end)
    elif custom != None:
        idx_range = list(custom)
    else:
        idx_range = range(dataset[deg].shape[0])
    plt.ion()
    for idx, vec in enumerate(dataset[deg]):
        if not idx in idx_range:
            continue
        print(idx, vec.shape)
        fig = plt.figure('Argument offset idx: %d, deg: %s'%(idx, deg), figsize=(200,6))
        sub = fig.add_subplot(1,1,1)
        sub.set_ylim(-3.5,3.5)
        title = ['ant2-ant1', 'ant3-ant1', 'ant4-ant1', 'ant3-ant2', 'ant4-ant2', 'ant4-ant3']
        for j in range(6):
            sub.plot(vec[j*52:(j+1)*52], '-*', label=title[j])
            sub.legend()
        plt.pause(1/freq)
        plt.close()

def sta_plot(dataset, deg=None, custom=None):
    if deg==None:
        print('Please choose the deg you want to plot!')
        return

    if custom != None:
        idx_range = custom
    else:
        idx_range = range(dataset[deg].shape[0])
    for idx, vec in enumerate(dataset[deg]):
        if not idx in idx_range:
            continue
        print(idx, vec.shape)
        plt.figure('Argument offset idx: %d, deg: %s'%(idx, deg), figsize=(200,6))
        plt.title('Phase offset concatnnate 6')
        plt.ylim(-3.5,3.5)
        title = ['ant2-ant1', 'ant3-ant1', 'ant4-ant1', 'ant3-ant2', 'ant4-ant2', 'ant4-ant3']
        for j in range(6):
            plt.plot(vec[j*52:(j+1)*52], '-*', label=title[j])
            plt.title('Phase offset concatnnate 6')
            plt.legend()
        plt.show()



def work(save=False, merge=False, avg=False):
    dataset = Data()#处理好的数据集为dataset
    if save:
        if merge:
            if os.path.exists(dataset_path):#save_path+'/offset_np'如果已经存在，则备份并重新创建一个offset_np
                i = 0
                while os.path.exists(dataset_path+'.bk'+str(i)):
                    i += 1
                subprocess.call('mv %s %s'%(dataset_path, dataset_path+'.bk'+str(i)), shell=True)
            subprocess.call('mkdir -p %s %s'%(dataset.train_path, dataset.test_path), shell=True)#如果不存在则创建该文件路径
        else:
            if os.path.exists(dataset_sep_path):#save_path+'/offset_np_each'如果已经存在，则备份并重新创建一个；如果不存在，则创建该文件路径
                i = 0
                while os.path.exists(dataset_sep_path+'.bk'+str(i)):
                    i += 1
                subprocess.call('mv %s %s'%(dataset_sep_path, dataset_sep_path+'.bk'+str(i)), shell=True)
            subprocess.call('mkdir -p %s %s'%(dataset.train_sep_path, dataset.test_sep_path), shell=True)


        info = dataset.info
        print("info is :",info)
        
        if avg:
            print('data_split: generating avg_arg for info...')
            for deg in info['degs']:
                for ant in list(info[deg].keys()):
                    if ant not in ants:
                        continue
                    for dis in list(info[deg][ant].keys()):
                        info[deg][ant][dis]['avg_arg'] = (info[deg][ant][dis]['arg']).replace('arg_np', 'arg_np-avg')
            


            print('data_split: finished. \nsaving info...')
            info_file = os.path.join(info_path, 'info.json')
            with open(info_file, 'w') as fp:
                json.dump(info, fp)
            print('data_split: info saved.')




        #print(info)
        if merge:
            dataset.load_arg_data(info, avg)
            dataset.gen_dataset()
            dataset.split_dataset(True)
        else:
            dataset.load_arg_data_each(info, avg)
            dataset.gen_dataset_each()
            dataset.split_dataset_each(True)
    else:
        info = dataset.info
        print("info is",info)
        if merge:
            dataset.load_dataset()
            # print(dataset.test_labels[:20])
        else:
            dataset.load_dataset_each()
            # dataset.plot_mean()
            dataset.plot_packet_mean()



    # workflow: load_info()->load_arg_data(info)->gen_dataset()->split_dataset(True)->Done
    # now just: load_dataset()

if __name__ == "__main__":
    # work(True)
    # work(save=False, merge=False)
    work(save=True, merge=True, avg=True)