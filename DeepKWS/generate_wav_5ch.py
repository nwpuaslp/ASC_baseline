# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University \
# (Authors: Zhanheng Yang)
import soundfile as sf

output_dir = './input_5ch/data/'

f = open('./input/wav.scp')
line = f.readline()

while line:
    w_name , w_dir = line.split()
    sig , rate = sf.read(w_dir)
    sig = sig[:,:5]
    sf.write(output_dir + w_name + '.wav' , sig , rate)

    print(w_name + ' done')
    line = f.readline()
