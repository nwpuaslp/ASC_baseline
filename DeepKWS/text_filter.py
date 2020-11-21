f = open('./feats/feats.scp')
line = f.readline()

ft = open('./input/text')
tline = ft.readline()

fw = open('./feats/text' , 'w')

utt_dict = {}

while line:
    w_name = line.split()[0]
    utt_dict[w_name] = 1
    line = f.readline()

while tline:
    t_name = tline.split()[0]
    if(t_name in utt_dict.keys()):
        fw.write(tline)
    tline = ft.readline()
    
