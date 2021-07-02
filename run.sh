#!/bin/bash

#audio_list=(1 3 5 7 9 11)
##audio_list=(1 3 7 9 11)
##audio_list=(1)
list=(10 30 50 80 100 200 500 1000 2000 5000 10000)
##list2=(20000 50000 100000 120000 150000)

#for i in ${audio_list[@]}; do
#    python feature_sentence.py $i
#    echo ""
#done

#for i in ${audio_list[@]}; do
#    python feature_speaker.py $i
#    echo ''
#done

for j in ${list[@]}; do
    python audit_speaker.py $j
    echo ''
done

#for i in ${audio_list[@]}; do
#    for j in ${list[@]}; do
#        python audit_speaker.py $i $j
#        echo ''
#    done
#done

#for i in ${audio_list[@]}; do
#    python plot_fig.py $i
#    echo ''
#done