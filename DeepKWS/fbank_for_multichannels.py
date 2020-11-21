# coding=UTF-8
# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University \
# (Authors: Zhanheng Yang)

from python_speech_features import fbank
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import random
import sys
import pdb

channels = 1

wav_dir = sys.argv[1]
save_dir = "./feats/data/"

def load_noise(path='dat/_background_noise_/'):
    noise = []
    files = os.listdir(path)    #返回path下文件列表
    for f in files:
        filename = f
        if ('wav' not in filename):     #跳过非wav文件
            continue
        f = os.path.join(path, f)
        sig, rate = sf.read(f)   #返回采样率和音频数据 stero: sig(n,2), mono:(n,)
        noise.append(sig)
    return  noise

def generate_fbank(sig, rate, noise=None, noise_weight=0.1, winlen=0.03125, winstep=0.03125/2, numcep=13, nfilt=26, nfft=512, lowfreq=20, highfreq=4000, winfunc=np.hanning, ceplifter=0, preemph=0.97):
    '''
    if(len(sig) != sig_len):
        if(len(sig)< sig_len):  #最后不足一帧，补0
            sig = np.pad(sig, (0, sig_len - len(sig)), 'constant')
        if(len(sig) >sig_len):
            sig = sig[0:sig_len]
    # i dont know, 'tensorflow' normalization
    sig = sig.astype('float') / 32768

    if(noise is not None):
        noise = noise[random.randint(0, len(noise)-1)] # pick a noise  随机选一种噪音
        start = random.randint(0, len(noise)-sig_len) # pick a sequence
        noise = noise[start:start+sig_len]
        noise = noise.astype('float')/32768
        sig = sig * (1-noise_weight) + noise * noise_weight     #加权直接加（对应采样点直接加）
        #wav.write('noise_test.wav', rate, sig)
    '''

    if channels == 1:
        fbank_feat , _ = fbank(sig , samplerate=rate, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                        highfreq=highfreq, winfunc=winfunc, preemph=preemph)
        fbank_feat = fbank_feat.astype('float32')
    else:
        fbank_feat = []
        for i in range(channels):
            sig_single = sig[: , i]
            fbank_feat_single , _ = fbank(sig_single , samplerate=rate, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                            highfreq=highfreq, winfunc=winfunc, preemph=preemph)
            fbank_feat_single = fbank_feat_single.astype('float32')
            fbank_feat.append(fbank_feat_single)

    return fbank_feat

def read_pcm(fileName):
    file = open(fileName, 'rb')
    base = 1 / (1<<15)
    
    shortArray = array.array('h') # int16
    size = int(os.path.getsize(fileName) / shortArray.itemsize)
    count = int(size / 2)
    shortArray.fromfile(file, size) # faster than struct.unpack
    file.close()
    shortArray = np.array(shortArray/32767)
    return shortArray


def merge_fbank_file(input_path='',mix_noise=False, sig_len=16000, winlen=0.025, winstep=0.01, nfilt=40, nfft=512,
                    lowfreq=20, highfreq=7600, winfunc=np.hanning, ceplifter=0, preemph=0.97):

    train_data = []

    train = open(wav_dir)
    train_line = train.readline()
    while(train_line):
        #print("1\n")
        #f = os.path.join(input_path, train_line[0:19])
        #print(f)
        f1 = train_line.strip('\n')
        flist = f1.split()
        if len(flist) > 2:
            print("warning")
            train_line = train.readline()
            continue
        sig, rate = sf.read(flist[1])
        #sig = sig[:,:1]
        #sig = sig/np.max(np.abs(sig))
        data = generate_fbank(sig, rate, winlen=winlen, winstep=winstep, 
                                nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                                highfreq=highfreq, winfunc=winfunc, ceplifter=ceplifter, preemph=preemph)
        data = np.array(data)
        #print(data.shape)
        if flist[0][-4] == ".wav" or flist[0][-4] == ".WAV":
            np.save(save_dir + flist[0][-4] + '.npy', data)
        else:
            np.save(save_dir + flist[0] + '.npy', data)
        print(flist[0] + ' done')

        train_line = train.readline()


if __name__ == "__main__":

    # test
    print("start")
    merge_fbank_file()

    #print('x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())
    print("done")
