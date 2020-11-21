# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University \
# (Authors: Zhanheng Yang)

import os
import tensorflow as tf
import numpy as np
import baseline_tdnn_inference as inference
import datetime
from tensorflow.python import pywrap_tensorflow
import soundfile as sf

MODEL_SAVE_PATH = "./baseline_tdnn_aec_bf_all"
num_type = 3
channels = 1
mfcc_dim = 40
maxT = 200
wav_maxT = 80000
batch_size = 1
win_smooth = 50
win_max = 100
win_int = 20
threshold = 0.30

feat_list_dir = "./feats/feats.scp"
text_list_dir = "./input/text"

def data_prepare_fbank():
    test_utt_num = 0
    ffeat = open(feat_list_dir)
    fline = ffeat.readline()

    ftext = open(text_list_dir)
    tline = ftext.readline()

    text_dict = {}
    while tline:
        t_name , t_dir = tline.split()
        text_dict[t_name] = t_dir
        tline = ftext.readline()

    data = []
    label = []
    length = []
    while fline:
        test_utt_num += 1
        flist = fline.split()

        x = np.load(flist[1])
        length.append(x.shape[0])
        #cmvn
        mean = np.mean(x , axis = 0)
        x[:] = x[:] - mean
        data.append(x)
        
        if text_dict[flist[0]] == "-1":
            label.append(0)
        else:
            label.append(1)
        
        fline = ffeat.readline()
 
    ffeat.close()

    return data , label , length , test_utt_num

def cal_cfd(y_output):
    cfd_collection = []
    for j in range(y_output.shape[0]):
        tem_0 = []
        tem_1 = []
        smo_p_0 = []
        smo_p_1 = []
        cfd = []
        predictions = y_output[j]
        for i in range(predictions.shape[0]):
            tem_0.append(predictions[i][0])
            tem_1.append(predictions[i][1])
            if i < win_smooth:
                smo_p_0.append(sum(tem_0)/(i+1))
                smo_p_1.append(sum(tem_1)/(i+1))
            else:
                smo_p_0.append(sum(tem_0[(i - win_smooth + 1):])/win_smooth)
                smo_p_1.append(sum(tem_1[(i - win_smooth + 1):])/win_smooth)
            temp = []
            if i < win_max:
                temp.append(max(smo_p_0))
                temp.append(max(smo_p_1))
                cfd_temp = (temp[0] * temp[1]) ** 0.5
            else:
                temp.append(max(smo_p_0[(i - win_max + 1):]))
                temp.append(max(smo_p_1[(i - win_max + 1):]))
                cfd_temp = (temp[0] * temp[1]) ** 0.5
            cfd.append(cfd_temp)
        cfd_collection.append(max(cfd))
    #print(max(cfd))
    return cfd_collection

def print_frr(cfd , label):
    fa_num = 0
    fr_num = 0
    f_num = 0
    t_num = 0

    for j in range(len(cfd)):
        if cfd[j] >= threshold and label[j] != 1:
            fa_num += 1 
        elif cfd[j] < threshold and label[j] != 0:
            fr_num += 1

        if label[j] == 0:
            f_num += 1
        elif label[j] == 1:
            t_num += 1
    
    print("frr: " + str(fr_num/t_num))



def test():
    gpu_options = tf.GPUOptions(allow_growth=True)

    def parser(record):
        features = tf.parse_single_example(
            record ,
            features = {
                'raw_x':tf.FixedLenFeature([] , tf.string),
                'raw_label':tf.FixedLenFeature([] , tf.int64)
            }
        )

        decode_x = tf.decode_raw(features['raw_x'] , tf.float64)
        decode_x = tf.reshape(decode_x , [80000])

        decode_y = features['raw_label']

        return decode_x , decode_y

    data , label , length_total , test_utt_num = data_prepare_fbank()

    x = tf.placeholder(tf.float32 , [None , None , mfcc_dim] , name = 'x-input')

    length = tf.placeholder(tf.float32 , [None] , name = 'length')
    y_pred = inference.inference(x , maxT, 1 , mfcc_dim , False)
    y_pred = tf.nn.softmax(y_pred)

    saver = tf.train.Saver(max_to_keep=30)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        saver.restore(sess , ckpt.model_checkpoint_path)

        cfd_fin = []

        for i in range(int(np.ceil(test_utt_num/batch_size))):
            time3 = datetime.datetime.now()
            start = batch_size * i
            end = batch_size * i + batch_size

            y_output = sess.run(y_pred , feed_dict={ x : data[start:end] , length : length_total[start:end]})

            output = cal_cfd(y_output)

            cfd_fin += output
            #print(len(label))
            print(output)
            time4 = datetime.datetime.now()

        print_frr(cfd_fin , label)   
        print('done')

        

def main():
    '''
    x_train = np.ones((100,maxT,channals,mfcc_dim))
    x_ref_train = np.ones((100,maxT,mfcc_dim))
    y_train = np.zeros((100,maxT,num_type))
    length_train = maxT - 10
    #x_test = np.ones((100,16,12,1))
    #y_test = np.zeros((100,3))
    x_val = np.ones((100,maxT,channals,mfcc_dim))
    x_ref_val = np.ones((100,maxT,mfcc_dim))
    y_val = np.zeros((100,maxT,num_type))
    length_val = maxT - 10
    '''

    test()


if __name__=='__main__':
    main()
