#!/Users/caiyunxiang/anaconda3/bin/python
# -*- coding: utf-8 -*-
# Processing collected data, remove noises, ready for svm/nn/gan train.
import re
import os
import numpy as np
import json
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from aoa_utils import *    # import global variables



pause = False
line = None
packet = None

cur_deg = None
cur_ant = None
cur_dis = None
PLOT = False
info = None

denoise_win_threshold = 3
#avg = True # average each packet through 68 symbols   这是个开关，是否对每个packet的68个symbols取平均


#function：获取所有文件路径
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


def load_data(target_paths):
    file_paths = get_file_paths(data_path)#把总的path传进去，得到file_path，可以依据角度和天线检索
    print('data_process load data: ', file_paths)#把所有路径输出
    data = {}  # format: data {'degs':['deg30', 'deg35', 'deg40'], 'deg30': deg_data1, 'deg35': deg_data2, ...}
    data['degs'] = []
    for deg in degs:
        global cur_deg
        cur_deg = deg
        data['degs'].append(deg)
        data[deg] = load_deg(deg, file_paths)#把数据读到data数组中，key为deg
    return data



#针对某个角度，遍历某条天线
def load_deg(deg, target_paths):
    file_paths = target_paths[deg]
    print('load_deg file_paths',file_paths)
    deg_data = {}  # format: {'10.2': ant_data, '20.3': ...}
    for ant in ants:
        global cur_ant
        cur_ant = ant
        deg_data[ant] = load_ant(ant, file_paths)
        # break
    return deg_data



def load_ant(ant, target_paths):
    file_paths = target_paths[ant]#该角度下，每一根天线的文件路径存在file_paths中
    print('load_ant file_paths',file_paths)
    ant_data = {}  # format: {'d3.54': file_data, 'd4.35': file_data, ...}
    for f in file_paths:
        dis = re.search('d\d\.\d+', f).group()
        global cur_dis
        cur_dis = dis
        # if dis != 'd4.35':
        #     continue
        if not PLOT:
            ant_data[dis] = read_file(f)
        else:
            # if info[cur_deg][ant][dis]['noise_idx'] == []:
                # continue
            # plot_read_file(f, info[cur_deg][ant][dis]['noise_idx']) #搞清楚几个plot分别绘制的是什么
            plot_read_file(f, list(range(int(info[cur_deg][ant][dis]['pkt_range'][1])+1)))
        # break
    return ant_data




def read_file(file_path, denoise_flag=True):
    print('reading file: ', file_path, '...')
    file_data = {} # format: {'pkt_range':['0', 'n'], 'arg': ndarray}
    file_data['pkt_range'] = ['0']
    file_data['noise_idx'] = []
    pkt_no = '0'
    arr = []
    temp = []
    infile = open(file_path)
    for line in infile.readlines():
        line = line.split()
        if(line[1] == '0'): # first symbol from a new packet
            pkt_no = line[0]
            # save current packet to tmp, call denoise


            temp.clear()  # ? may cause error. bug why not error?
            # update: I FIGURED OUT WHY! cause I used arr += temp, which is a copy operation!
            # if using arr.append(temp), when temp.clear() executed, arr would also be cleared!


            temp.append(line[2:])#把每个子载波的信道估计值相位
        else:
            temp.append(line[2:])
            if(line[1] == str(pakcet_len-1)):
                if(len(temp) != pakcet_len):
                    raise RuntimeError('Packet length is not %d'%pakcet_len)#如果不够68个symbol则报这个错误
                keep = True

                # the data of the first symbol is NOT CSI, we should drop them.第一个symbol不是CSI，需要丢弃掉
                temp = temp[1:]
                if denoise_flag:
                    keep = denoise(int(pkt_no), temp)  
                
                # denoise would drop all data, so we just skip it now...


                #if int(pkt_no) == 4 and cur_deg == 'deg55' and cur_ant == '20.3':
                #    keep = denoise(int(pkt_no), temp)
                #    if int(pkt_no) == 67:
                #        plot(int(pkt_no), temp)
                 #   plot(int(pkt_no), temp)
                #anima_plot_pkt(pkt_no, temp)
                if not keep:
                    file_data['noise_idx'].append(pkt_no)
                #if avg:
                    #temp = list(np.sum(np.array(temp, dtype='d'), axis=0))#这里
                arr += temp
        # arr.append(line[2:])
    if(pkt_no != '0'):
        file_data['pkt_range'].append(pkt_no)
    arr = np.array(arr, dtype='d')
    arr = np.delete(arr, 32, axis=1)
    file_data['arg'] = arr[:,6:58]
    return file_data




#Function:绘制读取文件的图
#Input: 文件路径，噪声列表
#Output:绘图，自变量为pk_no,因变量为每个包的CSI
def plot_read_file(file_path, noise_list = []):
    print('reading file: ', file_path, '...')
    # print('list: ', noise_list)
    pkt_no = '0'
    temp = []
    infile = open(file_path)
    for line in infile.readlines():
        line = line.split()
        if(line[1] == '0'): # first symbol from a new packet
            pkt_no = line[0]
            # save current packet to tmp, call denoise
            temp.clear()
            temp.append(line[2:])
        else:
            temp.append(line[2:])
            if(line[1] == str(pakcet_len-1)):
                if(len(temp) != pakcet_len):
                    raise RuntimeError('Packet length is not %d'%pakcet_len)
                if int(pkt_no) in noise_list:
                    plot(int(pkt_no), temp)
    return None





def denoise(pkt_idx, packet, show=False):
    # detect noise
    packet = np.array(packet, dtype='d')
    packet = np.delete(packet, 32, axis=1)
    packet = packet[:,6:58]
    keep = True
    win_size = 5
    threshold = 0.5
    noise_file = os.path.join(noise_path, 'denoise_%s_%s_%s.txt'%(cur_dis, cur_deg, cur_ant))
    mode = 'x'
    if os.path.exists(noise_file):
        mode = 'a'
    with open(noise_file, mode) as fp:
        fp.write('Packet number: %d:\n'%pkt_idx)
        sym_cnt = 0
        for idx in range(packet.shape[0]):
            fp.write('Symbol number: %d:\n'%idx)
            win_cnt = 0
            for beg in range(valid_carrier-win_size+1):
                var = np.var(packet[idx][beg:beg+win_size])
                if(var > threshold):
                    win_cnt += 1
                    # print('\tnoise in symbol %d in window %d detected, var: %f'%(idx, beg, var))
                    fp.write('\tnoise in symbol %d in window %d detected, var: %f\n'%(idx, beg, var))
                # else:
                #     fp.write('\tvar: %f\n'%var)
            if win_cnt >= denoise_win_threshold:
                sym_cnt += 1
                print('noise in symbol %d confirmed.'%idx)
                fp.write('-----noise in symbol %d confirmed.-----\n'%idx)
            if sym_cnt >= denoise_win_threshold:
                keep = False
                print("Drop packet %d."%pkt_idx)
                fp.write("-----Drop packet %d.-----\n"%pkt_idx)
                break
        # plot noise
        if show and not keep:
            print("Showing dropping noise:")
            #for idx in range(len(packet)):
                #plot_sym(pkt_idx, idx, packet[idx])
            #plot_pkt(pkt_idx, packet, freq=100)
        return keep




def plot(pkt_idx, packet):
    packet = np.array(packet, dtype='d')
    packet = np.delete(packet, 32, axis=1)
    packet = packet[:,6:58]
    # for idx in range(packet.shape[0]):
        # plot_sym(pkt_idx, idx, packet[idx])
    plot_pkt(pkt_idx, packet, freq=100)

### anination plot csi

def gen_sym_from_pkt():
    global packet
    for idx in range(packet.shape[0]):
        if not pause:
            yield idx, packet[idx]

def on_click(event):
    global pause
    pause ^= True

def sym_data(gen_sym_from_pkt):
    idx, csi = gen_sym_from_pkt[0], gen_sym_from_pkt[1]
    line.set_data(csi, idx)
    return line

def anima_plot_pkt(pkt_idx, pkt):
    pkt = np.array(pkt, dtype='d')
    global packet
    packet = pkt
     # title = ['symbol'+str(i) for i in range(pkt.shape[0])]
     # fig = plt.figure('Pakcet_idx: %s, Symbol_idx: %d'%(pkt_idx, idx), figsize=(200,6))
    fig = plt.figure('Pakcet_idx: %s'%pkt_idx, figsize=(200,6))
    sub = fig.add_subplot(1,1,1)
    sub.set_ylim(-3.2,3.2)
    sub.set_xlim(0, 64)
    global line
    line, = sub.plot([], [], '-*', ms=10)
    sub.legend()
    fig.canvas.mpl_connect('button_press_event', on_click)
    ani = animation.FuncAnimation(fig, sym_data, gen_sym_from_pkt, blit=False, interval=10, repeat=True)
    plt.show()

### animation plot csi finish

def plot_sym(pkt_idx, sym_idx, symbol):
    plt.figure('Packet_idx: %d, symbol_idx: %d'%(pkt_idx, sym_idx), figsize=(200,6))
    plt.title('CSI')
    plt.ylim(-3.2,3.2)
    plt.plot(symbol, '-*', label='CSI')
    plt.title('CSI')
    plt.legend()
    plt.show()

def plot_pkt(pkt_idx, pkt, freq=50):
    title = ['symbol'+str(i) for i in range(pkt.shape[0])]
    plt.ion()
    for idx, vec in enumerate(pkt):
        # print(idx, vec.shape)
        fig = plt.figure('Pakcet_idx: %d, Symbol_idx: %d'%(pkt_idx, idx), figsize=(200,6))
        sub = fig.add_subplot(1,1,1)
        sub.set_ylim(-3.2,3.2)
        sub.plot(vec, '-*', label=title[idx])
        sub.legend()
        plt.pause(1/freq)
        plt.close()

def store_data(data):
    denoise_log = os.path.join(noise_path, 'denoise_log.txt')
    with open(denoise_log, 'x') as logf:
        for deg in data['degs']:
            for ant in list(data[deg].keys()):
                for dis in list(data[deg][ant].keys()):
                    if dis not in data[deg].keys():
                        data[deg][dis] = {'noise_idx':set()}
                    print('%s, %s, %s, pkt_range: '%(deg, ant, dis), data[deg][ant][dis]['pkt_range'])
                    print('%s, %s, %s, pkt_range: '%(deg, ant, dis), data[deg][ant][dis]['pkt_range'], file=logf)
                    print("removed noises %s: "%ant, data[deg][ant][dis]['noise_idx'])
                    print("removed noises %s: "%ant, data[deg][ant][dis]['noise_idx'], file=logf)
                    data[deg][dis]['noise_idx'] |= set(data[deg][ant][dis]['noise_idx'])   # union
            for ant in list(data[deg].keys()):
                if ant not in ants:
                    continue
                for dis in list(data[deg][ant].keys()):
                    print(deg+', '+ant+', '+dis+', pkt_range: ', data[deg][ant][dis]['pkt_range'])
                    print('%s, %s, %s, pkt_range: '%(deg, ant, dis), data[deg][ant][dis]['pkt_range'])
                    print('%s, %s, %s, pkt_range: '%(deg, ant, dis), data[deg][ant][dis]['pkt_range'], file=logf)
                    data[deg][dis]['noise_idx'] = list(data[deg][dis]['noise_idx'])
                    print("removed noises total: ", data[deg][dis]['noise_idx'])
                    print("removed noises total: ", data[deg][dis]['noise_idx'], file=logf)
                    store_file = deg+'_'+dis+'_'+ant
                    store_file = os.path.join(store_path, store_file)
                    print(store_file, "original shape: ", data[deg][ant][dis]['arg'].shape)
                    print(store_file, "original shape: ", data[deg][ant][dis]['arg'].shape, file=logf)
                    # This is wrong. Deletion should be performed at the same time, or idx changed after each deletion.
                    # for pkg_idx in data[deg][dis]['noise_idx']:
                    #     idx = int(pkg_idx)
                    #     data[deg][ant][dis]['arg'] = np.delete(data[deg][ant][dis]['arg'], range(idx, idx+68), axis=0)

                    # The right way.
                    print('generating remove_list...')
                    remove_list = []
                    for pkg_idx in data[deg][dis]['noise_idx']:
                        idx = int(pkg_idx)
                        #if avg:#这里修改了
                            #remove_list.append(idx)#
                        #else:#
                        remove_list.append(list(range(idx*csi_num, idx*csi_num+csi_num)))#
                    print('ready to delete, remove_list: ', remove_list)
                    print('remove_list: ', remove_list, file=logf)
                    print('start deleting...')
                    data[deg][ant][dis]['arg'] = np.delete(data[deg][ant][dis]['arg'], remove_list, axis=0)
                    print(store_file, "denoised shape: ", data[deg][ant][dis]['arg'].shape)
                    print(store_file, "denoised shape: ", data[deg][ant][dis]['arg'].shape, file=logf)
                    np.save(store_file, data[deg][ant][dis]['arg'])
                    data[deg][ant][dis]['arg'] = store_file
    # store data info to json
    info_file = os.path.join(info_path, 'info.json')
    with open(info_file, 'x') as fp:
        json.dump(data, fp)#json存储位置信息
        print('info.json saved.')


#function:处理
#save=True ： 读取并存储数据
#save=False ： 读取数据，不存储数据    从info_file中读取info的json文件，令PLOT=True后执行load_data
def work(save=False):
    if save:
        for fd in global_paths:
            if os.path.exists(fd):#如果里面的内容存在了，就移动到.bk后面去
                i = 0
                while os.path.exists(fd+'.bk'+str(i)):#不覆盖掉上次的运行结果
                    i += 1
                subprocess.call('mv %s %s'%(fd, fd+'.bk'+str(i)), shell=True)
            subprocess.call('mkdir %s'%fd, shell=True)
        data = load_data(data_path)#读取数据
        # print('avg mode: average each packet through all 68 symbols.')
        print('storing...')
        s

        tore_data(data)#存储数据
        if data == None:#判断是否成功
            print('Loading failed.')
            return False
    else:
        info_file = os.path.join(info_path, 'info.json')#拼接路径
        with open(info_file, 'r') as fp:#打开一个读路径下的json文件，这个就是用来画图的？
            global info
            info = json.load(fp)     #这个生成的json文件的作用是个啥？
        print(info)
        global PLOT
        PLOT = True                  #将PLOT置为True后再load
        data = load_data(data_path)
    # to load npy, file path is os.path.join(data[deg][ant][dis]['arg'], '.npy')
    return True

if __name__ == "__main__":
    print("Loading data from ", data_path, "...")
    work(True)