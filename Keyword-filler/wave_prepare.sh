#!/bin/bash
data_dir=$1
prepare_pwd=$2


echo "start prepare"
echo $data_dir
echo $prepare_pwd

find $data_dir -name '*.wav' -print | xargs echo "" > $prepare_pwd/wav.scp.temp
OLD_IFS="$IFS"
IFS=" "
cat $prepare_pwd/wav.scp.temp | while read line
do
arr=($line);
IFS="$OLD_IFS"
for s in ${arr[@]};
do
        name=${s##*/}
        echo ${name%.*} ${s} >> $prepare_pwd/wav.scp_temp2;
done
done
rm $prepare_pwd/wav.scp.temp

sort -k 1 $prepare_pwd/wav.scp_temp2 > $prepare_pwd/wav.scp
rm $prepare_pwd/wav.scp_temp2
