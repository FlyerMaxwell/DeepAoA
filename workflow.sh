#!/usr/local/bin/bash

# workflow for processing raw data collected by USRP.

PROGRAM="$(basename $0)"
#osx
# usrp_data_src="/Volumes/My_Passport/Code/AoA/data/road-6-23-d3.04"
# ubuntu 
# usrp_data_src="/media/lion/My_Passport/Code/AoA/data/road-6-23-csi-raw/d5.23"
usrp_data_src="/Users/caiyunxiang/Desktop/AOA/6-15"
# mylib_path="/Users/yifengzhu/Code/mymods/my-lib"
mylib_path="/Users/caiyunxiang/AOA/mymods/my-lib"
# data_prc_path="/Users/yifengzhu/Code/AoA/code"
data_prc_path="/Users/caiyunxiang/Desktop/AOA/code"
log_file="$(basename $usrp_data_src)-log.txt"

function usage(){
    echo "$PROGRAM: usage: $PROGRAM [ -h | -i | p1 p2 p3 ]";
    return;
}


#将所有raw data的文件名输出到一个文件中
function usrp_data_list_gen(){
    if [[ -d "$usrp_data_src" ]]; then
        echo $usrp_data_src/*/raw-all | xargs -n1>$mylib_path/input_file_raw_all.txt;
    else
        echo "usrp_data_src does not exist.";
        return 1;
    fi
    return 0;
}

#将文件名作为参数传到read_all_grfile.app中，读取数据（app为c++编译后的文件）
function read_usrp_data(){
    $mylib_path/read_all_grfile.app<$mylib_path/input_file_raw_all.txt raw | tee $log_file
}


function data_process(){
    $data_prc_path/data_process.py>>$log_file;
}


function avg_arg_np(){
    $data_prc_path/avg_arg_np.py>>$log_file;
}

function data_split(){
    $data_prc_path/data_split.py>>$log_file;
}

function check_status(){
    if [[ $2 -ne 0 ]]; then
        echo "$1 failed, exit."
        exit 1;
    else
        echo "$1 finished."
    fi
}

function work(){
    usrp_data_list_gen;
    status=$?;
    check_status "usrp_data_list_gen" status ;#生成待处理数据文件名列表
    read_usrp_data;
    status=$?;
    check_status "read_usrp_data" status ;#读取二进制文件，转化为两种文件
    data_process;
    status=$?;
    check_status "data_process" status ;#生成npy
    avg_arg_np;
    status=$?;
    check_status "avg_arg_np" status ;
    data_split;
    status=$?;
    check_status "data_split" status ;#划分数据集
    exit 0;
}

usage;

if [[ $# -ne 3 ]]; then
    echo "3 parameters required, $# were given.";
    # usage;
    # exit 1;
    usrp_data_src=${1:-"$usrp_data_src"};
    mylib_path=${2:-"$mylib_path"};
    data_prc_path=${3:-"$data_prc_path"};
else
    usrp_data_src=$1;
    mylib_path=$2;
    data_prc_path=$3;
fi

# for p in $@; do
#     echo "$p";
# done

echo $usrp_data_src;
echo $mylib_path;
echo $data_prc_path;

work;
