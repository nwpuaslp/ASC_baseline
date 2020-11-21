#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import sys
import math

import torch as th
import torch.nn as nn
import torch.nn.parallel.data_parallel as data_parallel
import numpy as np
import librosa
import soundfile as sf

from feature_wj import FeatureExtractor, EPSILON, STFT, iSTFT

def read_scp(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def read_key(scp_file):
    scp_lst = read_scp(scp_file)
    buffer_key = []
    for line in scp_lst:
        key, path = line.strip().split()
        buffer_key.append(key)
    return buffer_key


def read_path(scp_file):
    scp_lst = read_scp(scp_file)
    buffer_path = []
    for line in scp_lst:
        key, path = line.strip().split()
        buffer_path.append(path)
    return buffer_path

def reload_for_eval(model, checkpoint_path, use_cuda):
    checkpoint = load_checkpoint(checkpoint_path, use_cuda)
    model.load_state_dict(checkpoint['model1'])
    print('=> Reload well-trained model {} for decoding.'.format(
            checkpoint_path))

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = th.load(checkpoint_path)
    else:
        checkpoint = th.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def evaluate(model, device, sets):
    acc15 = acc10 = acc5 = mae = 0
    model.eval()
    mix_scp = sets
    dataset = DataReader(mix_scp)
    total_num = len(dataset)
    print('=> Decoding ...')
    sys.stdout.flush()
    start_time = datetime.datetime.now()

    with th.no_grad():
        for idx, data in enumerate(dataset.read()):
            start = datetime.datetime.now()
            key = data['key']
            mix = data['mix'].to(device).float()
            doa = data['doa'].to(device).float()
            ssl, sns = data_parallel(model, (mix))
            speech_ssl = th.argmax(ssl*sns).item()
            if min(abs(int(speech_ssl)-doa),(360-abs(int(speech_ssl)-doa))) <= 10:
                acc15 = acc15+1
            if min(abs(int(speech_ssl)-doa),(360-abs(int(speech_ssl)-doa))) <= 7.5:
                acc10 = acc10+1
            if min(abs(int(speech_ssl)-doa),(360-abs(int(speech_ssl)-doa))) <= 5:
                acc5 = acc5+1
            mae = mae + min(abs(int(speech_ssl)-doa),360-abs(int(speech_ssl)-doa))
            elapsed = (datetime.datetime.now() - start).total_seconds()

        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print('=> Decode done. Total time is {:.2f} mins'.format(elapsed / 60.0))
        print('=> acc is {:.4f} {:.4f} {:.4f}'.format(acc15 / total_num, acc10 / total_num, acc5 / total_num))
        print(mae/total_num)

class DataReader(object):
    """Data reader for evaluation."""

    def __init__(self, mix_scp):
        self.key = read_key(mix_scp)
        self.mix_path = read_path(mix_scp)

    def __len__(self):
        return len(self.mix_path)

    def read(self):
        for i in range(len(self.mix_path)):
            key = self.key[i]
            mix = sf.read(self.mix_path[i])[0]
            max_norm = np.max(np.abs(mix[:,0]))
            mix = mix/max_norm
            mix = mix.transpose(1,0)
            mix = mix[:5]
            origin_length = mix.shape[0]
            doa = float(key.split('_')[-2])
            doa = np.array(doa)
            sample = {
                'key': key,
                'mix': th.from_numpy(mix.reshape(1,5,-1)),
                'origin_length':origin_length,
                'max_norm':max_norm,
                'doa':th.from_numpy(doa.reshape(1)),
            }
            yield sample

class Identity(nn.Module):
    def __init__(self,
                 in_channels=128,
                 conv_channels=256,
                 kernel_size=3,
                 dilation=1,
                 causal=True):
        super(Identity, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, conv_channels, (1, 1))
        self.identity_bn1 = nn.BatchNorm2d(conv_channels)
        self.identity_relu1 = nn.ReLU()
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv2d(
            conv_channels,
            conv_channels,
            (kernel_size,kernel_size),
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.identity_bn2 = nn.BatchNorm2d(conv_channels)
        self.identity_relu2 = nn.ReLU()
        self.sconv = nn.Conv2d(conv_channels, in_channels, (1, 1))
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad


    def forward(self, x):
        y = self.conv1x1(x)
        y = self.identity_bn1(self.identity_relu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad, :-self.dconv_pad]
        y = self.identity_bn2(self.identity_relu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class SLT_SSL_Baseline(nn.Module):
    def __init__(self,
                 frame_len=512,
                 frame_hop=256,
                 round_pow_of_two=True,
                 log_spectrogram=True,
                 mvn_spectrogram=True,
                 ipd_cos=True,
                 ipd_index="0,2;1,3;0,1"):
        super(SLT_SSL_Baseline, self).__init__()
        self.chunk_length = 7
        self.thereshold = 0.5
        self.doa_window = 20
        self.extractor = FeatureExtractor(frame_hop=frame_hop,
                                          frame_len=frame_len,
                                          normalize=True,
                                          round_pow_of_two=round_pow_of_two,
                                          log_spectrogram=log_spectrogram,
                                          mvn_spectrogram=mvn_spectrogram,
                                          ipd_cos=ipd_cos,
                                          ipd_index=ipd_index,
                                          ang_index="")
        # self.wpe = DNN_WPE()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(1,7), stride=(1,3))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1,5), stride=(1,2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.identity = nn.ModuleList()
        for i in range(3):
            self.identity.append(Identity(dilation=int(2**i)))
        self.sslconv1 = nn.Conv2d(in_channels=128, out_channels=360, kernel_size=(1,1))
        self.sslbn1 = nn.BatchNorm2d(360)
        self.sslrelu1 = nn.ReLU()
        self.snsconv1 = nn.Conv2d(in_channels=128, out_channels=360, kernel_size=(1,1))
        self.snsbn1 = nn.BatchNorm2d(360)
        self.snsrelu1 = nn.ReLU()
        self.sslconv2 = nn.Conv2d(in_channels=40, out_channels=1, kernel_size=(1,1))
        self.sslbn2 = nn.BatchNorm2d(1)
        self.sslrelu2 = nn.Sigmoid()
        self.snsconv2 = nn.Conv2d(in_channels=40, out_channels=1, kernel_size=(1,1))
        self.snsbn2 = nn.BatchNorm2d(1)
        self.snsrelu2 = nn.Sigmoid()
        self.MSELoss = nn.MSELoss()

    def reload_unmix(self):
        if self.unmix_cpt:
            sd = th.load(self.unmix_cpt, map_location="cpu")
            print(f"Reload unmix checkpoint from {self.unmix_cpt}...")
            self.unmix.load_state_dict(sd)

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

    def angle_loss(self, doa_angle, est_ssl, est_sns):
        """
        angle loss
        """
        
        def gaussian(mu,sigma):
            gs1 = th.arange(0, 720, device=est_sns.device)
            gs2 = th.arange(0, 720, device=est_sns.device)
            gs1 = th.exp(-(gs1-mu)**2/sigma**2)
            gs2 = th.exp(-(gs2-(mu+360))**2/sigma**2)
            if mu < 180:
                return th.max(gs1, gs2)[:360]
            else:
                return th.max(gs1, gs2)[360:720]

        def nearest_source_2mix(mu1, mu2):
            src = th.ones(360, device=est_sns.device)
            for i in range(360):
                if th.min(th.abs(i-mu1),th.abs(360-(i-mu1))) > th.min(th.abs(i-mu2),th.abs(360-(i-mu2))):
                    src[i]=0
            return src

        def nearest_source_3mix(mu1, mu2, mu3):
            src = th.ones(360, device=est_sns.device)
            for i in range(360):
                if th.min(th.abs(i-mu1),th.abs(360-(i-mu1))) > th.min(th.abs(i-mu3),th.abs(360-(i-mu3))) and th.min(th.abs(i-mu2),th.abs(360-(i-mu2))) > th.min(th.abs(i-mu3),th.abs(360-(i-mu3))):
                    src[i]=0
            return src

        def weighted_loss():
            ref_ssl = th.zeros(est_sns.shape, device=est_sns.device)
            ref_sns = th.zeros(est_sns.shape, device=est_sns.device)
            ref_w = th.zeros(est_sns.shape, device=est_sns.device)
            for i in range(doa_angle.shape[0]):
                ref_ssl[i] = th.max(gaussian(mu=doa_angle[i,0],sigma=45), gaussian(mu=doa_angle[i,1],sigma=45)) #node19ssl+sns:90 #node15sslg+sns01:45
                ref_sns[i] = nearest_source_2mix(mu1=doa_angle[i,0], mu2=doa_angle[i,1])
            return ref_ssl, ref_sns
        ref_ssl, ref_sns = weighted_loss()

        loss1 = self.MSELoss(est_ssl, ref_ssl)
        loss2 = self.MSELoss(est_sns, ref_sns)
        return loss1+loss2

    def forward(self, mix, ref=None, online=False):
        # mic = mix[:, :4].contiguous()
        mix_mag, mix_pha, _ = self.extractor(mix)
        N, C, D, T = mix_mag.shape
        output = th.cat([mix_mag, mix_pha], dim=1)
        output = output.transpose(3,2)
        output = self.conv1(output)
        output = self.relu1(self.bn1(output))
        output = self.conv2(output)
        output = self.relu2(self.bn2(output))
        for current_layer in self.identity:
            output = current_layer(output)
        ssl_output = self.sslrelu1(self.sslbn1(self.sslconv1(output)))
        sns_output = self.snsrelu1(self.snsbn1(self.snsconv1(output)))
        ssl_output = ssl_output.transpose(3,1)
        sns_output = sns_output.transpose(3,1)
        ssl_output = self.sslbn2(self.sslconv2(ssl_output))
        sns_output = self.snsbn2(self.snsconv2(sns_output))
        ssl_output = ssl_output.view([N, -1, 360])
        sns_output = sns_output.view([N, -1, 360])
        ssl_output = self.sslrelu2(th.mean(ssl_output, dim=1))
        sns_output = self.snsrelu2(th.mean(sns_output, dim=1))
        return ssl_output, sns_output


def main():
    device = th.device('cuda')
    model = SLT_SSL_Baseline()
    model.to(device)
    reload_for_eval(model, './model.pt', True)
    evaluate(model, device, './wav.scp')


if __name__ == '__main__':
    main()
