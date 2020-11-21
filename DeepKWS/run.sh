# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University \
# (Authors: Zhanheng Yang)

python generate_wav_5ch.py
./wave_prepare.sh ./input_5ch/data ./input_5ch
echo "generate 5ch done"

#AEC
python generate_list.py ./input_5ch/wav.scp
chmod 744 ./aec/run_aec.sh
./aec/run_aec.sh
./wave_prepare.sh $PWD/aec/data ./aec
echo "aec done"

#beamforming
cp ./aec/wav.scp ./bf/egs/ASC/wav.scp
cd ./bf
./run_ssl_beamforming_ssl_music.sh
cd ../
./wave_prepare.sh $PWD/bf/egs/ASC/data ./bf
echo "bf done"

#extract fbank
python fbank_for_multichannels.py ./bf/wav.scp
./feat_prepare.sh $PWD/feats/data ./feats
echo "fbank done"

#nn
python kws_test_ref_tdnn.py