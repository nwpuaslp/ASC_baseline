#/bin/bash
# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University \
# (Authors: Zhanheng Yang)
data_dir=./egs/ASC/data
egs_dir=ASC
# mkdir -p egs/$egs_dir/

# compute doa
#local/prepare.sh egs_dir egs/$egs_dir/data_mer
./scripts/run_ssl_music.sh --nj 100 --stft_conf conf/librosa_stft.conf --backend music --cmd "queue.pl" --doa-range 0,360 \
                   egs/$egs_dir egs/$egs_dir/sv.npy egs/$egs_dir/doa.scp

# beamforming
awk '{print $1,$2/1}' egs/$egs_dir/doa.scp > egs/$egs_dir/utt2beam
./scripts/run_fixed_beamformer.sh --nj 100 --cmd "queue.pl" --stft-conf conf/librosa_stft.conf \
                                --beam egs/$egs_dir/utt2beam \
                                egs/$egs_dir/wav.scp \
                                egs/$egs_dir/bf.npy \
                                $data_dir
                                # egs/$egs_dir/bf_result
# local/prepare.sh egs/$egs_dir/bf_result egs/$egs_dir/data_bf
