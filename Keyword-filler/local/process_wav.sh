#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University \
#                (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)

src=$1
src2=$2
trg=$3

rm input_5h/*.wav
rm input_5h/data/*
rm aec/data/*.wav
rm bf/egs/ASC/data/*.wav

mkdir -p ./input_5ch/data
python generate_wav_5ch.py $src ./input_5ch/data
python generate_wav_5ch.py $src2 ./input_5ch/data
./wave_prepare.sh ./input_5ch ./input_5ch/data
echo "generate 5ch done"

#AEC
python generate_list.py ./input_5ch/data/wav.scp
chmod 744 ./aec/run_aec.sh
./aec/run_aec.sh
./wave_prepare.sh $PWD/aec/data ./aec
echo "aec done"

#beamforming
cp ./aec/wav.scp ./bf/egs/ASC/wav.scp
cd ./bf
./run_ssl_beamforming_ssl_music.sh
cd ../
./wave_prepare.sh $PWD/bf/egs/ASC/data $trg
echo "bf done"

