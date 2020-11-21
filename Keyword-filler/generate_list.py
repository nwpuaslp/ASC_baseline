import sys

f = open(sys.argv[1])
line = f.readline()

fw = open("./aec/run_aec.sh" , 'w')

output_dir = "./aec/data/"

while line:
    wav_name , wav_dir = line.split()
    fw.write("./aec/process_aec_wukong " + wav_dir + ' ' + output_dir + wav_name + '.wav 1&\n')
    fw.write("sleep 0.05\n")
    line = f.readline()

f.close()
fw.close()

    
