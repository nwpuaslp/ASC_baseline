#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University(Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation(Authors:Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0

# get wav.scp utt2spk spk2utt here, we also just get _02_ _10_ mic because we don't need so much data

stage=0

data_base=$1
data_dev=$2
mkdir -p $data_dev
echo "prepare kws data"
if [ $stage -le 0 ];then
    find $data_base -name '*.wav' > $data_dev/wav.scp
fi

if [ $stage -le 1 ];then
    while read -r line;do
        line_p1=${line##*/}
        echo ${line_p1%.*} $line
    done < $data_dev/wav.scp  > $data_dev/wav_terminal.scp
    mv $data_dev/wav_terminal.scp $data_dev/wav.scp
fi

if [ $stage -le 2 ];then
    awk '{split($1,array,"_");print $1,array[1]}' $data_dev/wav.scp> $data_dev/utt2spk
    utils/utt2spk_to_spk2utt.pl $data_dev/utt2spk > $data_dev/spk2utt
    utils/fix_data_dir.sh $data_dev
fi
