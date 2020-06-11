import os
import json
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from aoa_utils import *    # import global variables

avg_arg_np_path = '/Users/caiyunxiang/Desktop/Self/Self_result/arg_np-avg'


subfiles = os.listdir(avg_arg_np_path)



def get_file_paths(data_path):
    subfolders = os.listdir(data_path)  #list所有该路径下的文件名
    paths = []
    file_paths = []
    for subfolder in subfolders:
        file_name = os.path.join(subfolder, 'raw-all')
        path = os.path.join(data_path, file_name)#获取到每一个raw-all路径并添加在paths里
        if (not os.path.isdir(path)):
            continue
        paths.append(path)

    for path in paths:
        files = os.listdir(path)#获取路径下的所有数据文件名
        for f in files:
            if ('arg' in f):#选择其中的相位数据
                file_path = os.path.join(path, f)
                file_paths.append(file_path)#拼接为文件路径，并加入到file_paths中
                print(file_path)#输出读取的相位数据文件

    target_paths = {} # {'deg30':{'10.2':[...], '10.4':[...], ...}, ...}
    for deg in degs:
        target_paths[deg] = {}
        for ant in ants:
            if (not (ant in target_paths[deg].keys())):
                target_paths[deg][ant] = []
            for f in file_paths:
                if deg in f and ant in f:
                    target_paths[deg][ant].append(f)
    return target_paths#把路径存在该变量中，key为deg和天线
#这里将路径下所有的目录全部存起来了











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
                    print("info[deg][ant][dis]['pkt_range'] is",info[deg][ant][dis]['pkt_range'])
                    if avg:
                        store_file = info[deg][ant][dis]['avg_arg']+'.npy'
                    else:
                        store_file = info[deg][ant][dis]['arg']+'.npy'
                    file_data = np.load(store_file)
                    if type(self.ant_data) == type(None):
                        self.ant_data = file_data
                    else:
                        self.ant_data = np.vstack((self.ant_data, file_data))
                    print("store_file is :", store_file, "file_data.shape is :",file_data.shape)
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

    def gen_dataset(self):
        """Generate offset dataset."""
        for deg in self.info['degs']:
            #if deg=='deg63':
              #  continue
            self.dataset[deg] = None
            for i, anti in enumerate(ants):
                for antj in ants[i+1:]:
                    print(anti, self.data[deg][anti].shape, antj, self.data[deg][antj].shape)
                    offset = self.data[deg][antj] - self.data[deg][anti]
                    if type(self.dataset[deg]) == type(None):
                        self.dataset[deg] = offset
                    else:
                        self.dataset[deg] = np.hstack((self.dataset[deg], offset))
            print(deg, self.dataset[deg].shape)
            print("saving dataset for ",deg, "...")
            np.save(os.path.join(dataset_path, deg), self.dataset[deg])
            print('dataset saved.')

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
            print(labels.shape, labels)
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


dataset = Data()#处理好的数据集为dataset
if os.path.exists(dataset_path):#save_path+'/offset_np'如果已经存在，则备份并重新创建一个offset_np
	i = 0
	while os.path.exists(dataset_path+'.bk'+str(i)):
		i += 1
	subprocess.call('mv %s %s'%(dataset_path, dataset_path+'.bk'+str(i)), shell=True)
    subprocess.call('mkdir -p %s %s'%(dataset.train_path, dataset.test_path), shell=True)#如果不存在则创建该文件路径

info = dataset.info
print("info is :",info)


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

dataset.load_arg_data(info, avg)
dataset.gen_dataset()
dataset.split_dataset(True)

