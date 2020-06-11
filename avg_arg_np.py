#!/Users/caiyunxiang/anaconda3/bin/python
# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import json
from aoa_utils import all_data_path, avg_store_path, csi_num

# work_path = '/Users/caiyunxiang/AOA/Data'
# data_path = work_path+'/road-6-15-data/d6.23-data/arg_np'
# store_path = work_path+'/road-6-15-data-avg/d6.23-data/arg_np-avg'

#def read_arg_np_files():


all_data_path = '/Users/caiyunxiang/Desktop/Self/Self_result/arg_np'

#comments:
#不要反复run data_process.py，超级浪费时间，直接修改路径run这个程序即可在对应位置生成arg_np_avg!!!!


files = os.listdir(all_data_path)
print("files",files)
print('Reading data from \'all_data_path:\'',all_data_path)
file_paths = []
for f in files:
    origin_path = os.path.join(all_data_path, f)
    dst_path = os.path.join(avg_store_path, f)
    file_paths.append((origin_path, dst_path))


print('file_paths is : ',file_paths)

for p,d in file_paths:
    print('processing', p)
    arg_np = np.load(p)
    rst = None
    for i in range(arg_np.shape[0]//csi_num):
        # print(f'i: {i}, array: {arg_np[i*csi_num:i*csi_num+csi_num][:,0]}')
        temp = np.sum(arg_np[i*csi_num:i*csi_num+csi_num],axis=0)/67
        # print(f'sum: {temp}')
        if type(rst) == type(None):
            rst = temp
        else:
            rst = np.vstack((rst, temp))
            # print(i)
    print(arg_np.shape, arg_np.shape[0]//csi_num)#对每次的67个CSI取平均   但是这里有个问题，你不是可能去掉其中一部分的symbol吗？还是说把相应的包都去掉而不是去掉symbols？
    print(rst.shape)
    np.save(d, rst)
    print(os.path.basename(d),' saved.')
    







    # break

#def work():
    #read_arg_np_files()

#if __name__ == "__main__":
#    work()