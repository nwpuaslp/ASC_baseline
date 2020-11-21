# Keyword Filler recipe
## Introducation
This script uses trained models to get the result of the development set. Using the script we can finally get the list of false alarm 
error rate and false reject error rate on each threshold between 0 and 1.

## Requirement
1. python3
2. soundfile0.10.6
3. scipy1.4.1
4. kaldi

## Usage
1. First, copy KWS-Dev.zip and SSL-Dev.zip into data/source/original

2. Second, use ln or cp to link or copy utils/ and steps/ in wsj/ kaldi recipe
```
ln -s ${KALDI_ROOT}/egs/wsj/s5/utils ${KEYWORD_FILLER_ROOT}/
ln -s ${KALDI_ROOT}/egs/wsj/s5/steps ${KEYWORD_FILLER_ROOT}/
```

3. Third, change the path.sh to find the kaldi root
```
export KALDI_ROOT=${KALDI_ROOT}
```
also you may not using kaldi python, you can add follow script at the end of path.sh
```
export PATH=${PYTHON_BIN}:$PATH
```

4. Forth, unzip exp.zip which is a trained kws model
```
unzip exp.zip
```

5. Finally, run the run_kws_kf.sh and you will get result.txt.
```
bash run_kws_kf.sh
```
