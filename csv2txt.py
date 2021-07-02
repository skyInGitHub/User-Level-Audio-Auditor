from __future__ import division
import os
import numpy as np
import pandas as pd
import argparse
import soundfile as sf
import python_speech_features as psf
import csv


def csv2txt(txt_memF, csv_memF):
    with open(txt_memF, "w") as output_file:
        with open(csv_memF, "r") as input_file:
            [output_file.write(row[0] + ' ' + row[1] + '\n') for row in csv.reader(input_file)]
        output_file.close()

    return 0


def csv2true(true_memF, csv_memF):
    with open(true_memF, "w") as output_file:
        with open(csv_memF, "r") as input_file:
            [output_file.write(row[0] + ' ' + row[2] + '\n') for row in csv.reader(input_file)]
        output_file.close()

    return 0


if __name__ == '__main__':

    working_dir = '/Users/skymiao/PycharmProjects/audio-auditor-extension/'

    # # Load csv
    csv_nonmemF = 'testing_auditor_100user/nonmember_test_clean_2_user.csv'
    csv_memF = 'testing_auditor_100user/member_train_clean_100_user.csv'
    # csv_mem = pd.read_csv(csv_memF)
    # csv_nonmem = pd.read_csv(csv_nonmemF)

    # # Save csv
    txt_nonmemF = "testing_auditor_100user/nonmember_test_clean_2_user.txt"
    txt_memF = "testing_auditor_100user/member_train_clean_100_user.txt"
    true_memF = "testing_auditor_100user/member_train_clean_100_user.txt"
    true_nonmemF = "testing_auditor_100user/nonmember_test_clean_2_user.txt"

    csv2txt(txt_memF, csv_memF)
    csv2true(true_memF, csv_memF)

    csv2txt(txt_nonmemF, csv_nonmemF)
    csv2true(true_nonmemF, csv_nonmemF)
