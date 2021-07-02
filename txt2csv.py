import sys
import os
import numpy as np
import pandas as pd
import argparse
import csv
import wave
import contextlib
import soundfile as sf
from pydub import AudioSegment


# Save csv file to output path
def array2csv(merge_array, csvfile):
    # Add header to final csv file
    # header = ['class', 's1', 'prob1', 'frame1', 's2', 'prob2', 'frame2', 's3', 'prob3', 'frame3',
    #           's4', 'prob4', 'frame4', 's5', 'prob5', 'frame5', 's6', 'prob6', 'frame6',
    #           's7', 'prob7', 'frame7', 's8', 'prob8', 'frame8']
    header = ['id', 'trans_txt', 'frame_len', 'txt', 'txt_len']

    # Save final array matrix to csv file with header defined
    pd.DataFrame(merge_array).to_csv(csvfile, header=header, index=None)


# Return merged array with label attached
def merge_test(matrix_in, matrix_in2, matrix_out):
    # Convert list[list] to array matrix
    merge_3 = matrix_in + matrix_in2 + matrix_out
    merge_array = np.row_stack(merge_3)

    # Replace the sixth column (sentence_id) with label ('member'/'non-member')
    for i in range(len(merge_array)):
        mem_len = len(matrix_in) + len(matrix_in2)
        if i < mem_len:
            merge_array[i, 5] = 'member'
        else:
            merge_array[i, 5] = 'nonmember'

    return merge_array


# Return merged array with label attached
def merge_train(matrix1_in, matrix2_in, matrix1_out, matrix1_out2, matrix2_out, matrix2_out2):
    # Merge list[list]s and convert the merged list[list] to array matrix
    merge_6 = matrix1_in + matrix2_in + matrix1_out + matrix1_out2 + matrix2_out + matrix2_out2
    merge_array = np.row_stack(merge_6)

    # Replace the first column (speaker_id) with label ('member'/'non-member')
    for i in range(len(merge_array)):
        if i < (len(matrix1_in)+len(matrix2_in)):
            merge_array[i, 0] = 'member'
        else:
            merge_array[i, 0] = 'nonmember'

    return merge_array


# Return merged array with label attached
def merge4array(matrix1, matrix2, matrix3, matrix4, label):
    # Merge list[list]s and convert the merged list[list] to array matrix
    merge_4 = matrix1 + matrix2 + matrix3 + matrix4
    merge_array = np.row_stack(merge_4)

    # Replace the first column (speaker_id) with label ('member'/'non-member')
    for i in range(len(merge_array)):
        merge_array[i, 0] = label

    return merge_array


# Return matrix which each row corresponding to one speaker
def matrix2spk(matrix, unique_list):
    matrix_spk = []     # Create an empty list for matrix_speaker
    spk_row = -1        # Row number for matrix_spk

    for uni_spk in unique_list:
        spk1_flg = 0            # Flag for each unique speaker's first record not found
        spk_row += 1            # Row number for matrix_spk
        matrix_spk.append([])   # Create a list for this speaker in matrix_spk list

        for row in range(len(matrix)):
            if uni_spk in matrix[row]:
                # If this unique speaker is found && its first record has not found
                if spk1_flg == 0:
                    spk1_flg = 1    # Flag for the unique speaker's 1st record has found

                    matrix_spk[spk_row].append(matrix[row][0])  # For the (spk_row)th speaker, add matched matrix(row).
                    matrix_spk[spk_row].append(matrix[row][1])
                    matrix_spk[spk_row].append(matrix[row][2])
                    matrix_spk[spk_row].append(matrix[row][3])

                elif spk1_flg == 1:
                    matrix_spk[spk_row].append(matrix[row][1])  # For this speaker, add match matrix[row] except [0] id
                    matrix_spk[spk_row].append(matrix[row][2])
                    matrix_spk[spk_row].append(matrix[row][3])

    if len(matrix_spk) == len(unique_list):
        print("Successfully merge each individual's multiple transcription recordings.")
        return matrix_spk
    else:
        print("Something wrong while merging each individual's transcription recordings.")
        sys.exit()


# Return unique list: refine unique item of the input list1
def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list


# Return matrix (list of list)
def txt2matrix(txtfile):
    # Open .txt file for operation
    with open(txtfile) as txt:
        row = 0
        nr = 0
        matrix = []
        for line in txt:
            word = line.split()

            if (row % 2) == 0:
                try:
                    word[1] = " ".join(word[1:len(word)])     # Merge all words into one string
                except IndexError as e:
                    print ("IndexError in {}'1 is not usual, previous id is {}".format(nr, matrix[nr-1][0]))
                    print (e)

                matrix.append([])               # Create a list in the list
                matrix[nr].append(word[0])      # Append sentence_id to this list of the list
                matrix[nr].append(word[1])      # Append predicted_txt (word type) to this list of the list
            else:
                try:
                    if matrix[nr][0] != word[0]:
                        print ("Error in {}'4 is not for {}".format(word[0], matrix[nr][0]))
                except IndexError as e:
                    print ("IndexError in {}'2 is not usual, previous id is {}".format(nr, matrix[nr - 1][0]))
                    print (e)
                # matrix[nr].append(word[2])      # Append probability to this list of the list
                matrix[nr].append(int(word[4]))      # Append frame_length to this list of the list
                nr += 1
            row += 1

    return matrix


# Return matrix with probability (list of list)
def txt2matrix_add(txtfile):
    # Open .txt file for operation
    with open(txtfile) as txt:
        row = 0
        nr = 0
        matrix = []
        for line in txt:
            word = line.split()

            if (row % 2) == 0:
                try:
                    word[1] = " ".join(word[1:len(word)])     # Merge all words into one string
                except IndexError as e:
                    print ("IndexError in {}'1 is not usual, previous id is {}".format(nr, matrix[nr-1][0]))
                    print (e)

                matrix.append([])               # Create a list in the list
                matrix[nr].append(word[0])      # Append sentence_id to this list of the list
                matrix[nr].append(word[1])      # Append predicted_txt (word type) to this list of the list
            else:
                try:
                    if matrix[nr][0] != word[0]:
                        print ("Error in {}'4 is not for {}".format(word[0], matrix[nr][0]))
                except IndexError as e:
                    print ("IndexError in {}'2 is not usual, previous id is {}".format(nr, matrix[nr - 1][0]))
                    print (e)
                matrix[nr].append(float(word[2]))      # Append probability to this list of the list
                matrix[nr].append(int(word[4]))      # Append frame_length to this list of the list
                nr += 1
            row += 1

    return matrix


# Return matrix (list of list)
def true2matrix(txtfile):
    # Open .txt file for operation
    with open(txtfile) as txt:
        nr = 0
        matrix = []
        for line in txt:
            word = line.split()
            word[1] = " ".join(word[1:len(word)])     # Merge all words into one string
            true_txt_len = len(word[1])          # Length of this sentence

            matrix.append([])               # Create a list in the list
            matrix[nr].append(word[0])      # Append sentence_id to this list of the list
            matrix[nr].append(word[1])      # Append sentence(str type) to this list of the list
            matrix[nr].append(true_txt_len)      # Append sentence_length to this list of the list
            nr += 1

    return matrix


# Return ted true_txt matrix (list of list)
def true2matrix_ted(txtfile):
    # Open .txt file for operation
    with open(txtfile) as txt:
        nr = 0
        matrix = []
        for line in txt:
            word = line.split()
            id_1 = word[3].split('.')
            id_2 = word[4].split('.')

            if word[3].find('.') == -1:
                id_1_uni = word[3] + '00'
            elif len(id_1[1]) == 1:
                id_1_uni = id_1[0] + id_1[1] + '0'
            else:
                id_1_uni = id_1[0] + id_1[1][0] + id_1[1][1]

            if word[4].find('.') == -1:
                id_2_uni = word[4] + '00'
            elif len(id_2[1]) == 1:
                id_2_uni = id_2[0] + id_2[1] + '0'
            else:
                id_2_uni = id_2[0] + id_2[1][0] + id_2[1][1]

            if len(id_1_uni) <= 7:
                n0 = 7 - len(id_1_uni)
                id_1 = '0' * n0 + id_1_uni
            else:
                print ("Error: user {} len(id_1_uni) = len({}) > 7".format(word[0], id_1_uni))
                exit(0)
            if len(id_2_uni) <= 7:
                n0 = 7 - len(id_2_uni)
                id_2 = '0' * n0 + id_2_uni
            else:
                print ("Error: user {} len(id_2_uni) = len({}) > 7".format(word[0], id_1_uni))
                exit(0)

            sen_id = word[0] + '-' + id_1 + '-' + id_2
            sentence = " ".join(word[6:len(word)])     # Merge all words into one string
            true_txt_len = len(sentence)          # Length of this sentence

            matrix.append([])               # Create a list in the list
            matrix[nr].append(sen_id)      # Append sentence_id to this list of the list
            matrix[nr].append(sentence)      # Append sentence(str type) to this list of the list
            matrix[nr].append(true_txt_len)      # Append sentence_length to this list of the list
            nr += 1

    return matrix


# Return matrix that each row corresponding to one sentence with
def matrix2sen(txt_sen, true_sen):
    # Check sum
    sen_all = 0          # Row number for matrix_sen
    matrix_sen = []     # Create an empty list for matrix_sentence

    # Check if it's true txt for this transcription result
    if len(txt_sen) == len(true_sen):
        print("Find the corresponding true txt.")
    else:
        print("Might Not the corresponding true txt. The len of txt_sen is {}, while true_sen is {}.".format(len(txt_sen), len(true_sen)))
        # sys.exit()

    for i in range(len(txt_sen)):
        sen_id_true = np.array(true_sen)[:,0].tolist()
        # Check id match or not, especially for ted
        if not txt_sen[i][0] in sen_id_true:
            temp = txt_sen[i][0]
            txt_sen_sp = txt_sen[i][0].split('-')

            txt_sen_id1 = str(int(txt_sen_sp[1]) + 1)
            n0 = 7 - len(txt_sen_id1)
            txt_sen_id1 = '0' * n0 + txt_sen_id1
            txt_sen1 = txt_sen_sp[0] + '-' + txt_sen_id1 + '-' + txt_sen_sp[2]

            txt_sen_id2 = str(int(txt_sen_sp[2]) + 1)
            n0 = 7 - len(txt_sen_id2)
            txt_sen_id2 = '0' * n0 + txt_sen_id2
            txt_sen2 = txt_sen_sp[0] + '-' + txt_sen_sp[1] + '-' + txt_sen_id2

            txt_sen3 = txt_sen_sp[0] + '-' + txt_sen_id1 + '-' + txt_sen_id2

            if txt_sen1 in sen_id_true:
                txt_sen[i][0] = txt_sen1
            elif txt_sen2 in sen_id_true:
                txt_sen[i][0] = txt_sen2
            elif txt_sen3 in sen_id_true:
                txt_sen[i][0] = txt_sen3
            else:
                print("ERROR: can't find user {}'s true_txt.".format(txt_sen[i][0]))
                sys.exit()
            print("Modify: user {} ---> {} = match true_txt.".format(temp, txt_sen[i][0]))

        for j in range(len(true_sen)):
            if true_sen[j][0] == txt_sen[i][0]:
                # If this unique sentence is found
                sen_all += 1

                txt_sen[i].append(true_sen[j][1])  # For the (i)th sentence, append matched true_sen[j][1]=true_txt.
                txt_sen[i].append(true_sen[j][2])  # append matched true_sen[j][2]=true_txt_length.
    print("SUCCESS: extract features for each sentence's transcription recording.")

    # txt_sen's current header is ['id', 'predicted_txt', 'frame_length', 'true_txt', 'true_txt_length']
    # change to matrix_spk's header = ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
    for k in range(len(txt_sen)):
        matrix_sen.append([])  # Create a list in the list
        matrix_sen[k].append(txt_sen[k][0])     # matrix_sen[j][0] = txt_sen[k][0] (sentence_id)
        matrix_sen[k].append(txt_sen[k][1])     # matrix_sen[j][0] = txt_sen[k][1] (predicted_txt)
        matrix_sen[k].append(txt_sen[k][3])     # matrix_sen[j][0] = txt_sen[k][3] (true_txt)
        matrix_sen[k].append(txt_sen[k][4])     # matrix_sen[j][0] = txt_sen[k][4] (true_txt_length)
        matrix_sen[k].append(txt_sen[k][2])     # matrix_sen[j][0] = txt_sen[k][2] (frame_length)

    return matrix_sen

    # return matrix_sen


# Return matrix with probability that each row corresponding to one sentence with
def matrix2sen_add(txt_sen, true_sen):
    # Check sum
    sen_all = 0          # Row number for matrix_sen
    matrix_sen = []     # Create an empty list for matrix_sentence

    # Check if it's true txt for this transcription result
    if len(txt_sen) == len(true_sen):
        print("Find the corresponding true txt.")
    else:
        print("Might Not the corresponding true txt. The len of txt_sen is {}, while true_sen is {}.".format(len(txt_sen), len(true_sen)))
        # sys.exit()

    for i in range(len(txt_sen)):
        sen_id_true = np.array(true_sen)[:,0].tolist()
        # Check id match or not, especially for ted
        if not txt_sen[i][0] in sen_id_true:
            temp = txt_sen[i][0]
            txt_sen_sp = txt_sen[i][0].split('-')

            txt_sen_id1 = str(int(txt_sen_sp[1]) + 1)
            n0 = 7 - len(txt_sen_id1)
            txt_sen_id1 = '0' * n0 + txt_sen_id1
            txt_sen1 = txt_sen_sp[0] + '-' + txt_sen_id1 + '-' + txt_sen_sp[2]

            txt_sen_id2 = str(int(txt_sen_sp[2]) + 1)
            n0 = 7 - len(txt_sen_id2)
            txt_sen_id2 = '0' * n0 + txt_sen_id2
            txt_sen2 = txt_sen_sp[0] + '-' + txt_sen_sp[1] + '-' + txt_sen_id2

            txt_sen3 = txt_sen_sp[0] + '-' + txt_sen_id1 + '-' + txt_sen_id2

            if txt_sen1 in sen_id_true:
                txt_sen[i][0] = txt_sen1
            elif txt_sen2 in sen_id_true:
                txt_sen[i][0] = txt_sen2
            elif txt_sen3 in sen_id_true:
                txt_sen[i][0] = txt_sen3
            else:
                print("ERROR: can't find user {}'s true_txt.".format(txt_sen[i][0]))
                sys.exit()
            print("Modify: user {} ---> {} = match true_txt.".format(temp, txt_sen[i][0]))

        for j in range(len(true_sen)):
            if true_sen[j][0] == txt_sen[i][0]:
                # If this unique sentence is found
                sen_all += 1

                txt_sen[i].append(true_sen[j][1])  # For the (i)th sentence, append matched true_sen[j][1]=true_txt.
                txt_sen[i].append(true_sen[j][2])  # append matched true_sen[j][2]=true_txt_length.
    print("SUCCESS: extract features for each sentence's transcription recording.")

    # txt_sen's current header is ['id', 'predicted_txt', 'frame_length', 'true_txt', 'true_txt_length']
    # change to matrix_spk's header = ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
    for k in range(len(txt_sen)):
        matrix_sen.append([])  # Create a list in the list
        matrix_sen[k].append(txt_sen[k][0])     # matrix_sen[j][0] = txt_sen[k][0] (sentence_id)
        matrix_sen[k].append(txt_sen[k][2])     # matrix_sen[j][0] = txt_sen[k][2] (probability)
        matrix_sen[k].append(txt_sen[k][1])     # matrix_sen[j][0] = txt_sen[k][1] (predicted_txt)
        matrix_sen[k].append(txt_sen[k][4])     # matrix_sen[j][0] = txt_sen[k][4] (true_txt)
        matrix_sen[k].append(txt_sen[k][5])     # matrix_sen[j][0] = txt_sen[k][5] (true_txt_length)
        matrix_sen[k].append(txt_sen[k][3])     # matrix_sen[j][0] = txt_sen[k][3] (frame_length)

    return matrix_sen

    # return matrix_sen


# Return matrix that each row corresponding to one speaker
def txt_matrix2spk(txtfile):
    matrix1 = txt2matrix(txtfile)

    multi_spk = []
    for row in matrix1:
        multi_spk.append(row[0])

    unique_list = unique(multi_spk)

    matrix_spk = matrix2spk(matrix1, unique_list)

    return matrix_spk


def audio_nframe(audio_f):
    # import wave
    # import contextlib
    # with contextlib.closing(open(audio_f, 'r')) as f:
    with contextlib.closing(wave.open(audio_f, 'r')) as f:
        n_frame = f.getnframes()
        rate = f.getframerate()
        duration = n_frame / float(rate)
        print(duration)

    # import soundfile as sf for flac
    f = sf.SoundFile(audio_f)
    print('samples = {}'.format(len(f)))
    print('sample rate = {}'.format(f.samplerate))
    print('seconds = {}'.format(len(f) / f.samplerate))

    return n_frame


def testf_user9():
    csvf_nonmem = "testing_auditor_100user/nonmember_test_clean_2_user.csv"
    csvf_nonmem1 = "testing_auditor_100user/nonmember9.csv"

    nonmem = pd.read_csv(csvf_nonmem)
    nonmem1 = pd.DataFrame(columns=nonmem.columns.values.tolist())
    n_audio = 9
    user_set = []
    user_N = 0

    for nrow in range(len(nonmem)):
        sen_id = nonmem.at[nrow, 'id'].split('-')
        user_id = sen_id[0]
        if user_id not in user_set:
            user_N += 1
            user_set.append(user_id)

    for user in user_set:
        n = 0
        for nrow in range(len(nonmem)):
            sen_id = nonmem.at[nrow, 'id'].split('-')
            user_id = sen_id[0]

            # frameLen = int(nonmem.at[nrow, 'frame_length'])
            # if user_id == user and frameLen <= 800:
            if user_id == user and n < n_audio:
                n += 1
                nonmem1 = nonmem1.append(nonmem.iloc[nrow])
        # if n < 5:
        #     print("User {} don't have 5 audios <= 800 frame_length".format(user))

    all = n_audio * user_N
    if len(nonmem1) != all:
        print ("Error: len(nonmem1) != all")

    # Some of them don't have 9 audios
    pd.DataFrame(nonmem1).to_csv(csvf_nonmem1, index=None)

    return 0


def testf_userAudio():
    audio_dir = '/Volumes/Sky Miao/nonmem/'
    csvf_nonmem1 = "testing_auditor_100user/nonmember5.csv"
    csvf_nonmem2 = "testing_auditor_100user/nonmember5_U52.csv"
    csvf_nonmem3 = "testing_auditor_100user/nonmember5_U52_2.csv"

    nonmem1 = pd.read_csv(csvf_nonmem1)
    length = 5 * 52
    nonmem2 = nonmem1[:length]
    pd.DataFrame(nonmem2).to_csv(csvf_nonmem2, index=None)

    column = ['id', 'frame_length']
    audio_frame = pd.DataFrame(columns=column)
    for audio_f in os.listdir(audio_dir):
        if audio_f.endswith(".flac"):
            audio_id = audio_f.split('.')[0]
            f = sf.SoundFile(audio_dir + audio_f)
            frame = len(f)
            audio_frame.loc[len(audio_frame)] = [audio_id, frame]

    # Merge two dataframe
    nonmem3 = nonmem2.drop('frame_length', axis=1)
    nonmem3 = pd.merge(nonmem3, audio_frame, on='id')

    # Some of them don't have 5 audios
    pd.DataFrame(nonmem3).to_csv(csvf_nonmem3, index=None)

    return 0


def get_arguments():
    parser = argparse.ArgumentParser(description='Description of your path of input and output files.')
    # parser.add_argument('in2', type=str, help='path to input in_file.txt')
    parser.add_argument('txtF', type=str, help='path to input file: trans_txt.txt')
    parser.add_argument('trueF', type=str, help='path to input file: true_txt.txt')
    # parser.add_argument('out3', type=str, help='path to input out_file.txt')
    # parser.add_argument('out4', type=str, help='path to input out_file.txt')
    parser.add_argument('csv', help='path of output file.csv')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    # txt_in = "testing_auditor_100user/member_train_clean_100_user.txt"
    # true_in = "testing_auditor_100user/train-clean-100-user-true-txt.txt"
    # csv_file = "testing_auditor_100user/member_train_clean_100_user_prob.csv"
    txt_in = "training_auditor_360shd_lstm/nonmember_test_clean_1_shd.txt"
    true_in = "training_auditor_360shd_lstm/test-clean-1-sh-true-txt.txt"
    csv_file = "training_auditor_360shd_lstm/nonmember_test_clean_1_shd_prob.csv"
    # # args = get_arguments()
    # # txt_in = args.txtF
    # # true_in = args.trueF
    # # csv_file = args.csv
    # txt_in = "training_auditor_timit_lstm/nonmember_test_timit.txt"
    # true_in = "training_auditor_timit_lstm/TEST-true-txt.txt"
    # csv_file = "training_auditor_timit_lstm/nonmember_test_timit.csv"
    print ("================================================")
    print (" Loading: txt_in =  {}".format(txt_in))
    print ("          true_in =  {}".format(true_in))
    print ("          csv_file =  {}".format(csv_file))
    print ("------------------------------------------------")
    print ("== START: 2 files loaded and output file defined. ")

    # # Convert txt_in (transcription results) to a matrix focusing on sentence id
    # txt_sen = txt2matrix(txt_in)
    # true_sen = true2matrix(true_in)
    # # true_sen = true2matrix_ted(true_in)
    # print ("The input txt_in obtains {} sentences.".format(len(txt_sen)))
    # print ("The input true_sen obtains {} sentences.".format(len(true_sen)))
    #
    # # Extract 4 features (trans_txt, true_txt, true_txt_length, frame_length) for each sentence id
    # txt_true_sen = matrix2sen(txt_sen, true_sen)
    # print ("The output {} file obtains {} sentences.".format(csv_file, len(txt_true_sen)))
    #
    # # Save as .csv focusing on sentence id
    # header = ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
    # pd.DataFrame(txt_true_sen).to_csv(csv_file, header=header, index=None)
    # print ("== END: save as {} file. ".format(csv_file))

    # Convert txt_in (transcription results) to a matrix with probability focusing on sentence id
    txt_sen = txt2matrix_add(txt_in)
    true_sen = true2matrix(true_in)
    print ("The input txt_in obtains {} sentences.".format(len(txt_sen)))
    print ("The input true_sen obtains {} sentences.".format(len(true_sen)))

    # Extract 5 features (probability, trans_txt, true_txt, true_txt_length, frame_length) for each sentence id
    txt_true_sen = matrix2sen_add(txt_sen, true_sen)
    print ("The output {} file obtains {} sentences.".format(csv_file, len(txt_true_sen)))

    # Save as .csv focusing on sentence id
    header = ['id', 'probability', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
    pd.DataFrame(txt_true_sen).to_csv(csv_file, header=header, index=None)
    print ("== END: save as {} file. ".format(csv_file))


    # args = get_arguments()
    #
    # txt1_in1 = args.in1
    # txt1_out1 = args.out1
    # txt1_out2 = args.out2
    # txt2_in1 = args.in2
    # txt2_out1 = args.out3
    # txt2_out2 = args.out4
    # csvfile = args.csv

    # # txt1_in1 = "data/test/test1_in.txt"
    # # txt1_out1 = "data/test/test1_out.txt"
    # # txt1_out2 = "data/test/test1_out2.txt"
    # txt1_in1 = "data/train/train1_in.txt"
    # txt1_out1 = "data/train/train1_out.txt"
    # txt1_out2 = "data/train/train1_out2.txt"
    # txt2_in1 = "data/train/train2_in.txt"
    # txt2_out1 = "data/train/train2_out.txt"
    # txt2_out2 = "data/train/train2_out2.txt"
    # # csvfile = "data/test/test.csv"
    # csvfile = "data/train/train.csv"

    # # if not os.path.isfile(txt1_in1 | txt1_out1 | txt1_out2):
    # #     print("File path {} or {} or {} does not exist. Exiting...".format(txt1_in1, txt1_out1, txt1_out2))
    # #     sys.exit()
    #
    # in_spk1 = txt_matrix2spk(txt1_in1)
    # out_spk1 = txt_matrix2spk(txt1_out1)
    # out_spk2 = txt_matrix2spk(txt1_out2)
    # in2_spk1 = txt_matrix2spk(txt2_in1)
    # out2_spk1 = txt_matrix2spk(txt2_out1)
    # out2_spk2 = txt_matrix2spk(txt2_out2)
    #
    # # array_merge = merge_test(in_spk1, out_spk1, out_spk2)
    # array_merge = merge_train(in_spk1, in2_spk1, out_spk1, out_spk2, out2_spk1, out2_spk2)
    #
    # array2csv(array_merge, csvfile)





