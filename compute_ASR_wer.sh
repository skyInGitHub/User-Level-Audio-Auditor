#!/bin/bash

## Convert xxx.log to xxx.csv
##
## Step1: xxx.log to xxx.txt ---> reduce some rebundent info
## Step2: select strings from xxx.txt to .csv
## Step2: formalize the xxx.csv with features


# Get input strings from cmd
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path/to/hyp_text> <path/to/ref_text>"
  echo "e.g: $0 nonmember_test_clean_2_user.txt test-clean-2-user-true-txt.txt"
  exit 1
fi

hyp_text_0=$1
ref_text=$2

txt_dir=$(echo $hyp_text_0| cut -d'.' -f 1)
tmp='_tmp.txt'
hyp_text=$txt_dir$tmp

awk 'NR%2==1' $hyp_text_0 > $hyp_text
compute-wer --text --mode=present ark:$ref_text ark:$hyp_text

echo " The END. "