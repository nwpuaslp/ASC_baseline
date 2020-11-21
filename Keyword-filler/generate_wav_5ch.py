import soundfile as sf
import argparse

parser = argparse.ArgumentParser(description="generate 5 channel wav")
parser.add_argument('input',help="input wav.scp")
parser.add_argument('output',help="output data")

FLAGS=parser.parse_args()
output_dir = FLAGS.output
f = open(FLAGS.input)
line = f.readline()

while line:
    w_name , w_dir = line.split()
    sig , rate = sf.read(w_dir)
    sig = sig[:,:5]
    sf.write(output_dir + w_name + '.wav' , sig , rate)

    print(w_name + ' done')
    line = f.readline()
