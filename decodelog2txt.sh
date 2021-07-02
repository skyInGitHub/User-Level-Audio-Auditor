#!/bin/bash

## Convert xxx.log to xxx.csv
##
## Step1: xxx.log to xxx.txt ---> reduce some rebundent info
## Step2: select strings from xxx.txt to .csv
## Step2: formalize the xxx.csv with features


# Get input strings from cmd
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path/to/dataset> <folder obtains logs>"
  echo "e.g: $0 data_libri/testing_set_for_auditor member_train"
  echo "e.g: $0 testing_set_for_auditor/decode_test_clean_2_user_out_dnn2 nonmember_test_clean_2_user"
  echo "e.g: $0 training_set_for_auditor/decode_test_clean_1_shd_out_dnn_2 nonmember_test_clean_1_shd"
  exit 1
fi

dataset=$1
label=$2

txt_dir=$(echo $dataset| cut -d'/' -f 1)        # txt_dir = data_libri (testing_set_for_auditor)
set_n=$(echo $dataset| cut -d'/' -f 2)          # set_n = testing_set_for_auditor (decode_test_clean_2_user_out_dnn2)
#set=$(echo $set_n| cut -d'_' -f 1)              # set = testing --- for "testing_set_for_auditor"
#set=$(echo $set_n| cut -d'_' -f 2)              # set = test --- for "decode_test_clean_2_user_out_dnn2"
#set=${set_n:0:5}

#if [ $label == "member" ]; then
#    txt_f=$txt_dir/member_$set.txt
#else
#    txt_f=$txt_dir/nonmember_$set.txt
#fi

#label_n=$(echo $label| cut -d'_' -f 1)
#txt_f=$txt_dir/$set-$label.txt                  # txt_f = data_libri/testing-member_train.txt
txt_f=$txt_dir/$label.txt                       # txt_f = testing_set_for_auditor/nonmember_test_clean_2_user.txt

#echo "=======  $(date) >> libri_$set.txt  ========="
#log_f=($(find $dataset/$label -type f -name "*.log" -print))
echo "=======  $(date) >> libri_$label.txt  ========="
log_f=($(find $dataset -type f -name "*.log" -print))

N_log=${#log_f[*]}
#N_log=1

echo " There are $N_log *.log files under $dataset/$label."
j=0
k=0
# In terminal, $log_f[0]=nothing; While run as ./decodelog2txt.sh, $log_g[8]=nothing
for ((i=0;i<$N_log;i++)); do
#  echo "=======  Data Process for $dataset/$label, ${log_f[$i]} to $txt_f ========="
  lines=$(wc -l ${log_f[$i]})
  echo "======= Data Process ${log_f[$i]} to $txt_f ========="

  sed -e 'h;s/.*utterance //' ${log_f[$i]} > ${log_f[$i]}.tmp
  tail -n +2 "${log_f[$i]}.tmp" > "${log_f[$i]}.tmp.tmp" && mv "${log_f[$i]}.tmp.tmp" "${log_f[$i]}.tmp"

  for ((t=1;t<=3;t++)); do
    sed -i '' -e '$ d' "${log_f[$i]}.tmp"
  done

  while IFS="" read -r p || [ -n "$p" ]; do
    k=$((k+1))
    printf '%s\n' "$p" >> "$txt_f"
  done < "${log_f[$i]}.tmp"

  lines2=$(wc -l ${log_f[$i]}.tmp)
  echo "$lines"
  echo "$lines2"

  rm "${log_f[$i]}.tmp"

  j=$((j+1))
done
#echo " There are $j (==$N_log) *.log files under $dataset/$label."
echo " There are $j (==$N_log) *.log files under $dataset."
echo " There are $k records in $txt_f."
echo " Check: $(wc -l $txt_f)."

echo " The END. "