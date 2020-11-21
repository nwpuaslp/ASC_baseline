#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0


# Download data 

data=$1
part=$2

if [ $# -ne 2 ];then
	echo "Usage: $0 <data-base> <corpus-part>"
	echo "e.g.: $0 data/aishell data_aishell.tar.gz"
fi

cd $data

if ! unzip $part;then
	echo "$0: error un-tarring archive $data/$part"
	exit 1;
fi

echo "$0: Successfully downloaded and un-tarred $data/$part"
exit 1;

