#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University \
#                (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
# Please put KWS-Dev.zip into the data/source first


data=data
# this path we use to put data source
data_source=data/source/original
# this path we use to put kaldi format data file
data_source_processed=data/source/processed
data_kws=data/kws/dev
data_neg=data/neg/dev
data_combine=data/combine/dev

ali=exp/datawukong_kws_spk_train_3hour_cut_fbank71_lats
feat_dir=data/fbank

kws_dict=data/dict

kws_word_split="悟空 悟空"
kws_word="悟空悟空"
kws_phone="w u4 k o1 ng w u4 k o1 ng"

use_fst=0

stage=5

. ./cmd.sh
. ./path.sh

# Prepare Data
if [ $stage -le 0 ]; then
	mkdir -p $data_kws
        mkdir -p $data_neg
        mkdir -p $data_combine
	mkdir -p $data_source_processed
	local/kws_untar.sh $data_source KWS-Dev.zip ;
        local/kws_untar.sh $data_source SSL-Dev.zip ;
	# You should write your own path to this script
	local/prepare_kws.sh $data_source/KWS-Dev $data_kws;
        local/prepare_kws.sh $data_source/SSL-Dev $data_neg
    local/process_wav.sh ${data_kws}/wav.scp ${data_neg}/wav.scp $data_combine ;
    awk '{print $1,$1}' $data_combine/wav.scp > $data_combine/utt2spk;
    awk '{print $1,"1"}' $data_kws/wav.scp > $data/label;
    awk '{print $1,"0"}' $data_neg/wav.scp >> $data/label;
    cp $data_combine/utt2spk $data_combine/spk2utt
    utils/fix_data_dir.sh data/combine/dev
fi

# Prepare keyword phone & id
if [ $stage -le 1 ];then
	[ ! -d $kws_dict ] && mkdir -p $kws_dict;
    echo "Prepare keyword phone & id"

	cat <<EOF > $kws_dict/lexicon.txt
sil sil 
<SPOKEN_NOISE> sil
<gbg> <GBG>
$kws_word $kws_phone
EOF
	cat <<EOF > $kws_dict/hotword.lexicon
$kws_word $kws_phone
EOF
	echo "<eps> 0
sil 1" > $kws_dict/phones.txt
	count=2
    awk '{for(i=2;i<=NF;i++){if(!match($i,"sil"))print $i}}' $kws_dict/lexicon.txt | sort | uniq  | while read -r line;do
		echo "$line $count"
		count=$(($count+1))
	done >> $kws_dict/phones.txt
	cat <<EOF > $kws_dict/words.txt
<eps> 0
<gbg> 1
$kws_word 2
EOF
fi

# Compile fst
if [ $stage -le 2 ]; then
	    echo "python local/gen_text_fst.py data/dict/hotword.lexicon data/dict/hotword.text.fst"
		python local/gen_text_fst.py data/dict/hotword.lexicon data/dict/fst.txt 
		fstcompile \
			--isymbols=data/dict/phones.txt \
			--osymbols=data/dict/words.txt  \
			data/dict/fst.txt | fstdeterminize | fstminimizeencoded > data/dict/hotword.openfst || exit 1;
		fstprint data/dict/hotword.openfst data/dict/hotword.fst.txt
fi

# Extracting feats
if [ $stage -le 3 ];then
	echo "Extracting feats & Create tr cv set"
    steps/make_fbank.sh --cmd "$train_cmd" --fbank-config conf/fbank71.conf --nj 1 $data_combine $data_combine/log $data_combine/fbank71 || exit 1;
    compute-cmvn-stats --binary=false --spk2utt=ark:$data_combine/spk2utt scp:$data_combine/feats.scp ark,scp:$data_combine/fbank71/cmvn.ark,$data_combine/fbank71/cmvn.scp || exit 1;
	utils/fix_data_dir.sh $data_combine
fi
# Testing
if [ $stage -le 4 ];then
	steps/nnet3/make_bottleneck_features.sh  \
		--use_gpu false \
		--nj 1 \
		output.log-softmax \
		$data_combine \
 		${data_combine}_bnf \
 		exp/nnet3/tdnn_test_kws \
 		exp/bnf/log \
 		exp/bnf || exit 1;
fi

# Decoding & Draw roc
if [ $stage -le 5 ];then
        utils/data/get_utt2dur.sh $data_combine
	copy-matrix scp:exp/bnf/raw_bnfeat_dev.1.scp t,ark:exp/bnf/ark.txt
	if [ $use_fst -eq 1 ];then
		python local/run_fst.py data/dict/hotword.fst.txt exp/bnf/ark.txt > result.txt
	else
		python local/kws_posterior_handling.py exp/bnf/ark.txt
	fi
	python local/kws_draw_roc.py --roc result.txt $data/label $data_combine/utt2dur
fi
