#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# global variables

#osx
# work_path = '/Volumes/My_Passport/Code/AoA/data'
#ubuntu
# work_path = '/media/lion/My_Passport/Code/AoA/data'
work_path = '/Users/caiyunxiang/Desktop/Self'#每次实验修改
# data_path = work_path+'/road-6-23-csi-raw/d5.23'
data_path = work_path+'/Self_data'	#每次实验修改
save_path = work_path+'/Self_result'			#每次实验修改

csi_path = save_path+'/csi'
store_path = save_path+'/arg_np'
info_path = save_path+'/data_info'
dataset_path = save_path+'/offset_np'
dataset_sep_path = save_path+'/offset_np_each'
noise_path = save_path+'/noise'


all_data_path = store_path
avg_store_path = save_path+'/arg_np-avg'

global_paths = [store_path, info_path, noise_path, avg_store_path]

#要处理的天线，这个不用修改
ants = ['10.2', '20.3', '20.5', '10.4']
#ants = ['10.2', '20.3', '10.4']
#ants = ['10.2']
#要处理的角度，这个需要每次进行修改
# degs = ['deg30', 'deg35', 'deg39', 'deg40', 'deg41', 'deg42', 'deg43']
# degs = ['deg'+str(i) for i in range(55,61)]
#degs = ['deg'+str(i) for i in range(55,71)]+['deg110','deg115','deg120']+['deg'+str(i) for i in range(125,129)]
#degs = ['deg55']

degs = ['deg'+str(i) for i in range(65,100,5)]+['deg'+str(i) for i in range(105,170,5)]
#degs = ['deg55']  # Test code  当只看某一天线和某一角度
#ants = ['10.2','20.3']  # Test code

unused = [str(i) for i in range(0,6)]+[str(32)]+[str(i) for i in range(59,64)]

sub_carrier = 64 # number of subcarriers in a symbol
valid_carrier = 52 # remove 0 subcarriers
pakcet_len = 68 # number of symbols in a packet
csi_num = 67 # drop the 1st, so 68-1 = 67
