from __future__ import division
import sys
import os
import re
from random import sample
import numpy as np
import pandas as pd
import argparse
import difflib
from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec, KeyedVectors
from statistics import mean, median, mode, stdev, variance
from gensim.test.utils import common_texts, get_tmpfile
import csv


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


# Return matrix which each word is processed with word embedding
def get_vector(model, word):
    # convert to lowercase, ignore all special characters - keep only
    # alpha-numericals and spaces
    word = re.sub(r'[^A-Za-z0-9\s]', r'', str(word).lower())

    #vectors = [model.wv[w] for w in word_tokenize(sentence) if w in model.wv]
    vectors = [model.wv[w] for w in word if w in model.wv]

    # # In case w not in sentence, change "vectors = [model.wv[w] for w in sentence if w in model.wv]"
    # vectors = []            # create a list
    # for w in sentence:
    #     if w in model.wv && w != '<UNK>':
    #         vectors.append(model.wv[w])
    #     else:
    #         seed = np.random.RandomState(0)
    #
    #         random_list = []
    #         for r in range(100):
    #             random_list.append(seed.uniform(-20, 20))
    #         vectors.append()

    v = np.zeros(model.vector_size)

    if (len(vectors) > 0):
        v = (np.array([sum(x) for x in zip(*vectors)])) / v.size

    return v


# Return two vector's similarity
def similarity(xv, yv):
    score = 0

    if xv.size > 0 and yv.size > 0:
        score = dot(xv, yv) / (norm(xv) * norm(yv))

    return score


# Return feats5 with two differences ==>> int vectors
def diff_chr2vec(feats5, csv, feats5_file):
    print ("-----------------------------------------")
    print ("Process: feats5 = ['diff_miss', 'diff_add'] str ==> nor(int_sum) ...")
    feats5_new = feats5

    # Search for each record
    for nrow in range(len(feats5)):
        int_miss = []
        int_add = []
        diff_miss = feats5.at[nrow, 'diff_miss']
        diff_add = feats5.at[nrow, 'diff_add']

        # Sum & Normalize the differences of miss chr (an integer representing the Unicode code)
        if diff_miss != '0' and isinstance(diff_miss, str):
            for w in diff_miss:
                int_miss.append(ord(w))
            nor_miss = sum(int_miss) / len(csv.at[nrow, 'true_text'])
        else:
            # if diff_miss != '0':
            #     print ("Warning: nrow = {} is diff_miss = {}".format(nrow, diff_miss))
            nor_miss = 0

        # Sum & Normalize the differences of add chr (an integer representing the Unicode code)
        if diff_add != '0' and isinstance(diff_add, str):
            for w in diff_add:
                int_add.append(ord(w))
            nor_add = sum(int_add) / len(csv.at[nrow, 'predicted_text'])
        else:
            # if diff_add != '0':
            #     print ("Warning: nrow = {} is diff_add = {}".format(nrow, diff_add))
            nor_add = 0

        feats5_new.at[nrow, 'diff_miss'] = nor_miss
        feats5_new.at[nrow, 'diff_add'] = nor_add

    # Write to csv file
    # feats5_f = feats5_file.split('_', 1)
    # feats5_new_f = feats5_f[0] + 'i_' + feats5_f[1]
    path1 = feats5_file.split('/')
    path2 = path1[2].split('_', 1)
    feats5_new_f = path1[0] + '/' + path1[1] + '/' + path2[0] + 'i_' + path2[1]
    try:
        feats5_new.to_csv(feats5_new_f, index=None)
    except IndexError as e:
        print e

    return feats5_new


# Return the differences (miss & add) of two strings
def chr_differ(txt_a, txt_b, diff_miss, diff_add, nrow):
    #   i is position; s[0]='-' means (txt_b less)=(txt_a added), s[0]='+' means (txt_b more)=(txt_a missed),
    #                  s[-1] presents changed character
    for i, s in enumerate(difflib.ndiff(txt_a, txt_b)):
        if s[0] == ' ':
            # txt_b = txt_a
            diff_miss[nrow] = diff_miss[nrow].__add__("")
            diff_add[nrow] = diff_add[nrow].__add__("")
        elif s[0] == '-':
            # txt_b - txt_a < 0 (s[-1] is chr txt_a added)
            diff_miss[nrow] = diff_miss[nrow].__add__("")
            diff_add[nrow] = diff_add[nrow].__add__(s[-1])
        elif s[0] == '+':
            diff_miss[nrow] = diff_miss[nrow].__add__(s[-1])
            diff_add[nrow] = diff_add[nrow].__add__("")

    return diff_miss[nrow], diff_add[nrow]


# Return the differences (miss & add) of two sub-strings
def iter_differ(txt_pre, txt_true, diff_miss, diff_add, nrow, iter_Flag):
    txt_diff_preA = []
    txt_diff_truB = []

    if len(txt_pre) <= 100 and len(txt_true) <= 100:
        [diff_miss[nrow], diff_add[nrow]] = chr_differ(txt_pre, txt_true, diff_miss, diff_add, nrow)
        iter_Flag = 1
        txt_diff_preA.append(txt_pre)
        txt_diff_truB.append(txt_true)

    else:
        txt_mat = difflib.SequenceMatcher(None, txt_pre, txt_true).get_matching_blocks()

        for i in range(len(txt_mat)):
            mati = txt_mat[i]
            if i == 0:
                txt_diff_preA.append(txt_pre[0:mati.a])
                txt_diff_truB.append(txt_true[0:mati.b])
            elif i > 0:
                matj = txt_mat[i - 1]
                txt_diff_preA.append(txt_pre[matj.a + matj.size:mati.a])
                txt_diff_truB.append(txt_true[matj.b + matj.size:mati.b])

        iter_Flag = 0


    return diff_miss[nrow], diff_add[nrow], txt_diff_preA, txt_diff_truB, iter_Flag


# Return statistics of a user' one feature (sum, max, min, avg, medium, standard deviation, variance)
def statistics(sentence_set):
    statics = []

    # sum, max, min, avg, medium, stdev, variance
    if len(sentence_set) == 1:
        statics.append(sentence_set[0])
        statics.append(sentence_set[0])
        statics.append(sentence_set[0])
        statics.append(sentence_set[0])
        statics.append(sentence_set[0])
        statics.append(sentence_set[0])
        statics.append(sentence_set[0])
    else:
        statics.append(sum(sentence_set))
        statics.append(max(sentence_set))
        statics.append(min(sentence_set))
        statics.append(mean(sentence_set))
        statics.append(median(sentence_set))
        statics.append(stdev(sentence_set))
        statics.append(variance(sentence_set))

    return statics


# Initial the list for processing original 4 feats into 3 feats except 'id'
# Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length'] = csv
#           ==> ['id', 'similarity', 'frame_length', 'speed']
# Return & Write feats3
def feats3(csv, feats_file, w2vModel):
    print ("-----------------------------------------")
    print ("Process: feats3 = ['id', 'similarity', 'frame_length', 'speed'] ...")
    feats3 = []
    for nrow in range(len(csv)):
        feats3.append([])  # Create a feats3 list in the list
        predicted_vectors = []
        true_vectors = []
        similarity_sum = 0

        predicted_text = csv.at[nrow, 'predicted_text'].split()
        true_text = csv.at[nrow, 'true_text'].split()

        # Condition 1: some words not predicted as real sentence
        #          ==> # of words in predicted text < # of words in true text
        if len(predicted_text) < len(true_text):
            for nw in range(len(predicted_text)):
                true_word = get_vector(w2vModel, true_text[nw])
                true_vectors.append(true_word)

                # vector for unknown predicted text
                if predicted_text[nw] == '<UNK>':
                    predicted_word = np.random.uniform(low=-50, high=50, size=w2vModel.vector_size)
                else:
                    predicted_word = get_vector(w2vModel, predicted_text[nw])

                predicted_vectors.append(predicted_word)
                similarity_sum += similarity(predicted_word, true_word)

            for nw in range(len(predicted_text), len(true_text)):
                true_word = get_vector(w2vModel, true_text[nw])
                true_vectors.append(true_word)

                predicted_word = np.random.uniform(low=-50, high=50, size=w2vModel.vector_size)
                predicted_vectors.append(predicted_word)

                similarity_sum += similarity(predicted_word, true_word)

        # Condition 2: words are predicted correctly in numbers as its true sentence
        #          ==> # of words in predicted text = # of words in true text
        elif len(predicted_text) == len(true_text):
            for nw in range(len(predicted_text)):
                true_word = get_vector(w2vModel, true_text[nw])
                true_vectors.append(true_word)

                if predicted_text[nw] == '<UNK>':
                    predicted_word = np.random.uniform(low=-50, high=50, size=w2vModel.vector_size)
                else:
                    predicted_word = get_vector(w2vModel, predicted_text[nw])

                predicted_vectors.append(predicted_word)
                similarity_sum += similarity(predicted_word, true_word)

        # Condition 3: one word not predicted correctly as two/more words
        #          ==> # of words in predicted text > # of words in true text
        else:
            for nw in range(len(true_text)):
                true_word = get_vector(w2vModel, true_text[nw])
                true_vectors.append(true_word)

                if predicted_text[nw] == '<UNK>':
                    predicted_word = np.random.uniform(low=-50, high=50, size=w2vModel.vector_size)
                else:
                    predicted_word = get_vector(w2vModel, predicted_text[nw])

                predicted_vectors.append(predicted_word)
                similarity_sum += similarity(predicted_word, true_word)

            for nw in range(len(true_text), len(predicted_text)):
                true_word = np.random.uniform(low=-50, high=50, size=w2vModel.vector_size)
                true_vectors.append(true_word)

                if predicted_text[nw] == '<UNK>':
                    predicted_word = np.random.uniform(low=-50, high=50, size=w2vModel.vector_size)
                else:
                    predicted_word = get_vector(w2vModel, predicted_text[nw])

                predicted_vectors.append(predicted_word)
                similarity_sum += similarity(predicted_word, true_word)

        # Complete the processed 3 features list for 1 id
        feats3[nrow].append(csv.at[nrow, 'id'])
        # feats3[nrow].append(csv.at[nrow, 'probability'])
        similarity_score = similarity_sum / len(true_text)
        feats3[nrow].append(similarity_score)
        feats3[nrow].append(csv.at[nrow, 'frame_length'])
        # speed = csv.at[nrow, 'true_text_length'] / csv.at[nrow, 'frame_length']
        speed = csv.at[nrow, 'frame_length'] / csv.at[nrow, 'true_text_length']
        feats3[nrow].append(speed)

    if len(feats3) != len(csv):
        print ("ERROR: some sentence id' similarity score didn't count into the similarity array.")
        sys.exit(1)
    else:
        print ("SUCCESS: complete {} sentences' similarity score and speed.".format(len(feats3)))

    # Save initial features as .csv focusing on sentence id
    # similarity = similarity_array
    # header = ['id', 'probability', 'similarity', 'frame_length', 'speed']
    header = ['id', 'similarity', 'frame_length', 'speed']
    pd.DataFrame(feats3).to_csv(feats_file, header=header, index=None)
    print ("Success: save as {}".format(feats_file))

    return pd.DataFrame(feats3, columns=header)


# Process the list of 3 feats into 5 feats except 'id'
# Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length'] = csv
# #         ==> ['id', 'similarity', 'frame_length', 'speed'] = feats3
#           ==> ['id', 'similarity', 'diff_miss', 'diff_add', 'frame_length', 'speed'] = feats5
# Return feats5
def feats5(csv, feats3, feats5_file):
    print ("-----------------------------------------")
    print ("Process: add ['diff_miss', 'diff_add'] into feats3 ...")
    diff_miss = []
    diff_add = []

    # For each records in csv process csv.at[nrow, 'predicted_text'] and csv.at[nrow, 'true_text']
    for nrow in range(len(csv)):
        diff_miss.append("")
        diff_add.append("")
        txt_pre = csv.at[nrow, 'predicted_text'].replace('<UNK> ', '')
        txt_pre = txt_pre.replace(' <UNK>', '')
        txt_true = csv.at[nrow, 'true_text']

        # Enumerate the differences
        if len(txt_pre) <= 100 and len(txt_true) <= 100:
            [diff_miss[nrow], diff_add[nrow]] = chr_differ(txt_pre, txt_true, diff_miss, diff_add, nrow)
        else:
            txt_diff_preA = []
            txt_diff_truB = []
            mat_txt = []

            # Get common substrings of txt_pre, txt_true ==> txt_mat (aggregate from each 100 length)
            seg = max(int(len(txt_pre)/100), int(len(txt_true)/100))
            for ind in range(seg):
                if (ind + 1) == seg:
                    ind_s = ind * 100
                    mat_tmp = difflib.SequenceMatcher(None, txt_pre[ind_s:len(txt_pre)],
                                                      txt_true[ind_s:len(txt_true)]).get_matching_blocks()
                    # Save Match.a, Match.b, Match.size to mat_txt for the last 100 length
                    for i in range(len(mat_tmp)):
                        a = mat_tmp[i].a + ind_s
                        b = mat_tmp[i].b + ind_s
                        mat_txt.append({'a': a, 'b': b, 'size': mat_tmp[i].size})
                else:
                    ind_s = ind * 100
                    ind_e = (ind + 1) * 100
                    mat_tmp = difflib.SequenceMatcher(None, txt_pre[ind_s:ind_e],
                                                      txt_true[ind_s:ind_e]).get_matching_blocks()
                    # Save Match.a, Match.b, Match.size to mat_txt for the middle 100 lengths
                    mat_tmp.pop()
                    for i in range(len(mat_tmp)):
                        a = mat_tmp[i].a + ind_s
                        b = mat_tmp[i].b + ind_s
                        mat_txt.append({'a': a, 'b': b, 'size': mat_tmp[i].size})
            # Convert common substring info into DataFrame with header = ['a', 'b', 'size']
            mat_txt = pd.DataFrame(mat_txt)

            # List different substrings of txt_pre, txt_true ==> txt_diff_preA, txt_diff_truB
            for i in range(len(mat_txt)):
                mati_a = mat_txt.at[i, 'a']
                mati_b = mat_txt.at[i, 'b']
                if i == 0:
                    txt_diff_preA.append(txt_pre[0:mati_a])
                    txt_diff_truB.append(txt_true[0:mati_b])
                elif i > 0:
                    matj_a = mat_txt.at[i-1, 'a']
                    matj_b = mat_txt.at[i-1, 'b']
                    matj_size = mat_txt.at[i-1, 'size']
                    txt_diff_preA.append(txt_pre[matj_a + matj_size:mati_a])
                    txt_diff_truB.append(txt_true[matj_b + matj_size:mati_b])

            # Loop the different substrings and add missed/added character into [diff_miss, diff_add]
            for m in range(len(txt_diff_preA)):
                if txt_diff_preA[m] != '' or txt_diff_truB[m] != '':
                    # At least one of substrings =\= empty set
                    if len(txt_diff_preA[m]) <= 100 and len(txt_diff_truB[m]) <= 100:
                        [diff_miss[nrow], diff_add[nrow]] = chr_differ(txt_diff_preA[m], txt_diff_truB[m],
                                                                       diff_miss, diff_add, nrow)
                    elif txt_diff_preA[m] == '' or txt_diff_truB[m] == '':
                        [diff_miss[nrow], diff_add[nrow]] = chr_differ(txt_diff_preA[m], txt_diff_truB[m],
                                                                       diff_miss, diff_add, nrow)
                    else:
                        [diff_miss[nrow], diff_add[nrow]] = chr_differ(txt_diff_preA[m], txt_diff_truB[m],
                                                                       diff_miss, diff_add, nrow)
                        print("The different substrings are over 100 length when nrow = {}:\n m={}, "
                              "len(txt_diff_preA[m])={}, len(txt_diff_truB[m]={}".format(nrow, m, len(txt_diff_preA[m]),
                                                                                         len(txt_diff_truB[m])))

        # If there are no missed/added characters, put 0 into that cell
        if len(diff_miss[nrow]) == 0:
            diff_miss[nrow] = 0
        elif len(diff_miss[nrow]) >= 50:
                print ("Alert: length of diff_miss[{}] = {} >= 50".format(nrow, len(diff_miss[nrow])))

        if len(diff_add[nrow]) == 0:
            diff_add[nrow] = 0
        elif len(diff_add[nrow]) >= 50:
                print ("Alert: length of diff_add[{}] = {} >= 50".format(nrow, len(diff_add[nrow])))


    # Check
    if len(diff_miss) != len(csv) or len(diff_add) != len(csv):
        print ("length of diff_miss, diff_add, csv: {}, {}, {}".format(len(diff_miss), len(diff_add), len(csv)))
        print ("ERROR: some sentence's predicted missed/added strings' combination didn't count.")
        sys.exit(1)

    # Add 2 additional feats to previous feats3 ==> feats5
    # feats3.insert(2, "diff_miss", diff_miss)
    # feats3.insert(3, "diff_add", diff_add)
    feats3.insert(3, "diff_miss", diff_miss)
    feats3.insert(4, "diff_add", diff_add)

    pd.DataFrame(feats3).to_csv(feats5_file, index=None)
    print ("Check feats5's header with 1 row:\n{}".format(feats3.head(1)))
    print ("Success: save as {}".format(feats5_file))
    print ("-----------------------------------------")

    return feats3


# Statistically analyze the list for processing 3 feats towards each user where 'id' = user#-chapter#-sentence#
# Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
#           ==> ['user', 'similarity_statistics', 'frame_length_statistics', 'speed_statistics']
# Return feats3_user
def feats_user3(feats_csv, feats_user_file):
    print ("-----------------------------------------")
    print ("Process: record to user feats3U3 = ['user', 'similarity_sta~', 'frame_length_sta~', 'speed_sta~'] ...")
    feats3_user = []
    user_set = []
    user_N = 0
    for nrow in range(len(feats_csv)):
        # Search user's id for each sentence
        sen_id = feats_csv.at[nrow, 'id'].split('-')
        user_id = sen_id[0]

        # Check new users
        if user_id not in user_set:
            # Set variables for each user
            feats3_user.append([])  # Create a feats3 list in the list
            sim_user = []
            frame_user = []
            speed_user = []

            # Search all sentences for this user
            for mrow in range(len(feats_csv)):
                sen_id_m = feats_csv.at[mrow, 'id'].split('-')
                user_id_m = sen_id_m[0]

                # Check new sentence for this user
                if user_id_m == user_id:
                    sim_user.append(feats_csv.at[mrow, 'similarity'])
                    frame_user.append(feats_csv.at[mrow, 'frame_length'])
                    speed_user.append(feats_csv.at[mrow, 'speed'])

            # Features statistics for all sentences of each new user
            sim_statics = statistics(sim_user)
            frame_statics = statistics(frame_user)
            speed_statics = statistics(speed_user)

            # Summarize the statistics for each new user
            feats3_user[user_N].append(user_id)
            for ss in sim_statics:
                feats3_user[user_N].append(ss)
            for fs in frame_statics:
                feats3_user[user_N].append(fs)
            for sps in speed_statics:
                feats3_user[user_N].append(sps)

            # Update user's number and user_set
            user_N += 1
            user_set.append(user_id)

    print ("SUCCESS: complete {} speakers' statistic features of similarity, frame length and speed.".format(user_N))

    # Save processed features as .csv focusing on user(speaker) id
    # similarity = similarity_array
    header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
              'similarity_stdev', 'similarity_var', 'frame_len_sum', 'frame_len_max', 'frame_len_min', 'frame_len_avg',
              'frame_len_medium', 'frame_len_stdev', 'frame_len_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg',
              'speed_medium', 'speed_stdev', 'speed_var']
    # header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
    #           'similarity_stdev', 'similarity_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg',
    #           'speed_medium', 'speed_stdev', 'speed_var']
    pd.DataFrame(feats3_user).to_csv(feats_user_file, header=header, index=None)
    print ("END: save as {}".format(feats_user_file))

    return pd.DataFrame(feats3_user, columns=header)


# Statistically analyze the list for processing 2 feats towards each user where 'id' = user#-chapter#-sentence#
# Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
#           ==> ['user', 'similarity_statistics', 'speed_statistics']
# Return feats3_user
def feats_user2(feats_csv, feats_user_file):
    print ("-----------------------------------------")
    print ("Process: record to user feats3U2 = ['user', 'similarity_sta~', 'speed_sta~'] ...")
    feats3_user = []
    user_set = []
    user_N = 0
    for nrow in range(len(feats_csv)):
        # Search user's id for each sentence
        sen_id = feats_csv.at[nrow, 'id'].split('-')
        user_id = sen_id[0]

        # Check new users
        if user_id not in user_set:
            # Set variables for each user
            feats3_user.append([])  # Create a feats3 list in the list
            sim_user = []
            frame_user = []
            speed_user = []

            # Search all sentences for this user
            for mrow in range(len(feats_csv)):
                sen_id_m = feats_csv.at[mrow, 'id'].split('-')
                user_id_m = sen_id_m[0]

                # Check new sentence for this user
                if user_id_m == user_id:
                    sim_user.append(feats_csv.at[mrow, 'similarity'])
                    frame_user.append(feats_csv.at[mrow, 'frame_length'])
                    speed_user.append(feats_csv.at[mrow, 'speed'])

            # Features statistics for all sentences of each new user
            sim_statics = statistics(sim_user)
            # frame_statics = statistics(frame_user)
            speed_statics = statistics(speed_user)

            # Summarize the statistics for each new user
            feats3_user[user_N].append(user_id)
            for ss in sim_statics:
                feats3_user[user_N].append(ss)
            # for fs in frame_statics:
            #     feats3_user[user_N].append(fs)
            for sps in speed_statics:
                feats3_user[user_N].append(sps)

            # Update user's number and user_set
            user_N += 1
            user_set.append(user_id)

    print ("SUCCESS: complete {} speakers' statistic features of similarity and speed.".format(user_N))

    # # Save processed features as .csv focusing on user(speaker) id
    # # similarity = similarity_array
    # header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
    #           'similarity_stdev', 'similarity_var', 'frame_len_sum', 'frame_len_max', 'frame_len_min', 'frame_len_avg',
    #           'frame_len_medium', 'frame_len_stdev', 'frame_len_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg',
    #           'speed_medium', 'speed_stdev', 'speed_var']
    header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
              'similarity_stdev', 'similarity_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg',
              'speed_medium', 'speed_stdev', 'speed_var']
    pd.DataFrame(feats3_user).to_csv(feats_user_file, header=header, index=None)
    print ("END: save as {}".format(feats_user_file))

    return pd.DataFrame(feats3_user, columns=header)


# Statistically analyze the list for processing 5 feats towards each user where 'id' = user#-chapter#-sentence#
# Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
#           ==> ['user', 'similarity_sta~', 'miss_sta~', 'add_sta~', 'frame_length_sta~', 'speed_sta~']
# Return feats5_user
def feats_user5(csv, feats5_csv, feats_user_file, feats5_file):
    print ("-----------------------------------------")
    print ("Process: record to user feats5U5 = ['user', 'similarity_sta~', 'miss_sta~', 'add_sta~', "
           " 'frame_length_sta~', 'speed_sta~'] ...")
    feats5_user = []
    user_set = []
    user_N = 0

    # Convert str(diff_miss, diff_add) to int_sum(diff_miss, diff_sum)
    # add csv[] ==> normalized_int_sum
    feats5 = diff_chr2vec(feats5_csv, csv, feats5_file)

    for nrow in range(len(feats5)):
        # Search user's id for each sentence
        sen_id = feats5_csv.at[nrow, 'id'].split('-')
        user_id = sen_id[0]

        # Check new users
        if user_id not in user_set:
            # Set variables for each user
            feats5_user.append([])  # Create a feats5 list in the list
            prob_user = []
            sim_user = []
            miss_user = []
            add_user = []
            frame_user = []
            speed_user = []

            # Search all sentences for this user
            for mrow in range(len(feats5_csv)):
                sen_id_m = feats5_csv.at[mrow, 'id'].split('-')
                user_id_m = sen_id_m[0]

                # Check new sentence for this user
                if user_id_m == user_id:
                    # prob_user.append(feats5_csv.at[mrow, 'probability'])
                    sim_user.append(feats5_csv.at[mrow, 'similarity'])
                    miss_user.append(feats5_csv.at[mrow, 'diff_miss'])
                    add_user.append(feats5_csv.at[mrow, 'diff_add'])
                    frame_user.append(feats5_csv.at[mrow, 'frame_length'])
                    speed_user.append(feats5_csv.at[mrow, 'speed'])

            # Features statistics for all sentences of each new user
            if len(sim_user) <= 1:
                print ("When nrow={}, user_id={} has only {} sentence".format(nrow, user_id, len(sim_user)))
            # prob_statics = statistics(prob_user)
            sim_statics = statistics(sim_user)
            miss_statics = statistics(miss_user)
            add_statics = statistics(add_user)
            frame_statics = statistics(frame_user)
            speed_statics = statistics(speed_user)

            # Summarize the statistics for each new user
            feats5_user[user_N].append(user_id)
            # for pp in prob_statics:
            #     feats5_user[user_N].append(pp)
            for ss in sim_statics:
                feats5_user[user_N].append(ss)
            for ms in miss_statics:
                feats5_user[user_N].append(ms)
            for ads in add_statics:
                feats5_user[user_N].append(ads)
            for fs in frame_statics:
                feats5_user[user_N].append(fs)
            for sps in speed_statics:
                feats5_user[user_N].append(sps)

            # Update user's number and user_set
            user_N += 1
            user_set.append(user_id)

    print ("SUCCESS: complete {} speakers' statistic features of similarity, frame length and speed.".format(user_N))

    # Save processed features as .csv focusing on user(speaker) id
    # similarity = similarity_array
    # header = ['user', 'prob_sum', 'prob_max', 'prob_min', 'prob_avg', 'prob_medium', 'prob_stdev', 'prob_var',
    #           'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
    #           'similarity_stdev', 'similarity_var', 'miss_sum', 'miss_max', 'miss_min', 'miss_avg', 'miss_medium',
    #           'miss_stdev', 'miss_var', 'add_sum', 'add_max', 'add_min', 'add_avg', 'add_medium', 'add_stdev',
    #           'add_var', 'frame_len_sum', 'frame_len_max', 'frame_len_min', 'frame_len_avg',
    #           'frame_len_medium', 'frame_len_stdev', 'frame_len_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg',
    #           'speed_medium', 'speed_stdev', 'speed_var']
    header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
              'similarity_stdev', 'similarity_var', 'miss_sum', 'miss_max', 'miss_min', 'miss_avg', 'miss_medium',
              'miss_stdev', 'miss_var', 'add_sum', 'add_max', 'add_min', 'add_avg', 'add_medium', 'add_stdev',
              'add_var', 'frame_len_sum', 'frame_len_max', 'frame_len_min', 'frame_len_avg',
              'frame_len_medium', 'frame_len_stdev', 'frame_len_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg',
              'speed_medium', 'speed_stdev', 'speed_var']
    pd.DataFrame(feats5_user).to_csv(feats_user_file, header=header, index=None)
    print ("END: save as {}".format(feats_user_file))

    return pd.DataFrame(feats5_user, columns=header)


# Statistically analyze the list for processing 4 feats towards each user where 'id' = user#-chapter#-sentence#
# Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
#           ==> ['user', 'similarity_sta~', 'miss_sta~', 'add_sta~', 'speed_sta~']
# Return feats4_user
def feats_user4(csv, feats5_csv, feats_user_file, feats5_file):
    print ("-----------------------------------------")
    print ("Process: record to user feats5U4 = ['user', 'similarity_sta~', 'miss_sta~', 'add_sta~', 'speed_sta~'] ...")
    feats5_user = []
    user_set = []
    user_N = 0

    # Convert str(diff_miss, diff_add) to int_sum(diff_miss, diff_sum)
    # add csv[] ==> normalized_int_sum
    feats5 = diff_chr2vec(feats5_csv, csv, feats5_file)

    for nrow in range(len(feats5)):
        # Search user's id for each sentence
        sen_id = feats5_csv.at[nrow, 'id'].split('-')
        user_id = sen_id[0]

        # Check new users
        if user_id not in user_set:
            # Set variables for each user
            feats5_user.append([])  # Create a feats5 list in the list
            sim_user = []
            miss_user = []
            add_user = []
            frame_user = []
            speed_user = []

            # Search all sentences for this user
            for mrow in range(len(feats5_csv)):
                sen_id_m = feats5_csv.at[mrow, 'id'].split('-')
                user_id_m = sen_id_m[0]

                # Check new sentence for this user
                if user_id_m == user_id:
                    sim_user.append(feats5_csv.at[mrow, 'similarity'])
                    miss_user.append(feats5_csv.at[mrow, 'diff_miss'])
                    add_user.append(feats5_csv.at[mrow, 'diff_add'])
                    frame_user.append(feats5_csv.at[mrow, 'frame_length'])
                    speed_user.append(feats5_csv.at[mrow, 'speed'])

            # Features statistics for all sentences of each new user
            sim_statics = statistics(sim_user)
            miss_statics = statistics(miss_user)
            add_statics = statistics(add_user)
            # frame_statics = statistics(frame_user)
            speed_statics = statistics(speed_user)

            # Summarize the statistics for each new user
            feats5_user[user_N].append(user_id)
            for ss in sim_statics:
                feats5_user[user_N].append(ss)
            for ms in miss_statics:
                feats5_user[user_N].append(ms)
            for ads in add_statics:
                feats5_user[user_N].append(ads)
            # for fs in frame_statics:
            #     feats5_user[user_N].append(fs)
            for sps in speed_statics:
                feats5_user[user_N].append(sps)

            # Update user's number and user_set
            user_N += 1
            user_set.append(user_id)

    print ("SUCCESS: complete {} speakers' statistic features of similarity, frame length and speed.".format(user_N))

    # Save processed features as .csv focusing on user(speaker) id
    # similarity = similarity_array
    # header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
    #           'similarity_stdev', 'similarity_var', 'miss_sum', 'miss_max', 'miss_min', 'miss_avg', 'miss_medium',
    #           'miss_stdev', 'miss_var', 'add_sum', 'add_max', 'add_min', 'add_avg', 'add_medium', 'add_stdev',
    #           'add_var', 'frame_len_sum', 'frame_len_max', 'frame_len_min', 'frame_len_avg', 'frame_len_medium',
    #           'frame_len_stdev', 'frame_len_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg', 'speed_medium',
    #           'speed_stdev', 'speed_var']
    header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
              'similarity_stdev', 'similarity_var', 'miss_sum', 'miss_max', 'miss_min', 'miss_avg', 'miss_medium',
              'miss_stdev', 'miss_var', 'add_sum', 'add_max', 'add_min', 'add_avg', 'add_medium', 'add_stdev',
              'add_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg', 'speed_medium', 'speed_stdev', 'speed_var']
    pd.DataFrame(feats5_user).to_csv(feats_user_file, header=header, index=None)
    print ("END: save as {}".format(feats_user_file))

    return pd.DataFrame(feats5_user, columns=header)


# Statistically analyze the list for processing 5 feats towards each user where 'id' = user#-chapter#-sentence#
# Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
#           ==> ['user', 'similarity_sta~', 'miss_sta~', 'add_sta~', 'frame_length_sta~', 'speed_sta~']
# Return feats5_user
def feats_user5_audio(csv, feats5_csv, feats5_Afile, feats5_file, n_audio):
    print ("-----------------------------------------")
    print ("Process: record to user feats5U5A{} = ['user', 'similarity_sta~', 'miss_sta~', 'add_sta~', "
           " 'frame_length_sta~', 'speed_sta~'] ...".format(n_audio))
    feats5_user = []
    user_set = []
    user_N = 0

    # Convert str(diff_miss, diff_add) to int_sum(diff_miss, diff_sum)
    # add csv[] ==> normalized_int_sum
    feats5 = diff_chr2vec(feats5_csv, csv, feats5_file)

    for nrow in range(len(feats5)):
        # Search user's id for each sentence
        sen_id = feats5_csv.at[nrow, 'id'].split('-')
        user_id = sen_id[0]

        # Check new users
        if user_id not in user_set:
            # Set variables for each user
            feats5_user.append([])  # Create a feats5 list in the list
            sim_user = []
            miss_user = []
            add_user = []
            frame_user = []
            speed_user = []

            # Search all sentences for this user
            for mrow in range(len(feats5_csv)):
                sen_id_m = feats5_csv.at[mrow, 'id'].split('-')
                user_id_m = sen_id_m[0]

                # Check new sentence for this user
                if user_id_m == user_id:
                    sim_user.append(feats5_csv.at[mrow, 'similarity'])
                    miss_user.append(feats5_csv.at[mrow, 'diff_miss'])
                    add_user.append(feats5_csv.at[mrow, 'diff_add'])
                    frame_user.append(feats5_csv.at[mrow, 'frame_length'])
                    speed_user.append(feats5_csv.at[mrow, 'speed'])

            # Features statistics for a few sentences of each new user
            if len(sim_user) < n_audio:
                print ("user_id={} has only {} sentence".format(user_id, len(sim_user)))
                sim_user = np.random.choice(sim_user, n_audio)
                miss_user = np.random.choice(miss_user, n_audio)
                add_user = np.random.choice(add_user, n_audio)
                frame_user = np.random.choice(frame_user, n_audio)
                speed_user = np.random.choice(speed_user, n_audio)
            else:
                sim_user = sample(sim_user, n_audio)
                miss_user = sample(miss_user, n_audio)
                add_user = sample(add_user, n_audio)
                frame_user = sample(frame_user, n_audio)
                speed_user = sample(speed_user, n_audio)

            sim_statics = statistics(sim_user)
            miss_statics = statistics(miss_user)
            add_statics = statistics(add_user)
            frame_statics = statistics(frame_user)
            speed_statics = statistics(speed_user)

            # Summarize the statistics for each new user
            feats5_user[user_N].append(user_id)
            for ss in sim_statics:
                feats5_user[user_N].append(ss)
            for ms in miss_statics:
                feats5_user[user_N].append(ms)
            for ads in add_statics:
                feats5_user[user_N].append(ads)
            for fs in frame_statics:
                feats5_user[user_N].append(fs)
            for sps in speed_statics:
                feats5_user[user_N].append(sps)

            # Update user's number and user_set
            user_N += 1
            user_set.append(user_id)

    print ("SUCCESS: complete {} speakers' statistic features of similarity, frame length and speed.".format(user_N))

    # Save processed features as .csv focusing on user(speaker) id
    # similarity = similarity_array
    header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
              'similarity_stdev', 'similarity_var', 'miss_sum', 'miss_max', 'miss_min', 'miss_avg', 'miss_medium',
              'miss_stdev', 'miss_var', 'add_sum', 'add_max', 'add_min', 'add_avg', 'add_medium', 'add_stdev',
              'add_var', 'frame_len_sum', 'frame_len_max', 'frame_len_min', 'frame_len_avg',
              'frame_len_medium', 'frame_len_stdev', 'frame_len_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg',
              'speed_medium', 'speed_stdev', 'speed_var']
    # header = ['user', 'similarity_sum', 'similarity_max', 'similarity_min', 'similarity_avg', 'similarity_medium',
    #           'similarity_stdev', 'similarity_var', 'speed_sum', 'speed_max', 'speed_min', 'speed_avg',
    #           'speed_medium', 'speed_stdev', 'speed_var']
    pd.DataFrame(feats5_user).to_csv(feats5_Afile, header=header, index=None)
    print ("END: save as {}".format(feats5_Afile))

    return pd.DataFrame(feats5_user, columns=header)


def get_arguments():
    parser = argparse.ArgumentParser(description='Description of your path of input and output files.')
    parser.add_argument('n_audio', type=int, help='# of querying audios')
    # parser.add_argument('out3', type=str, help='path to input out_file.txt')
    # parser.add_argument('out4', type=str, help='path to input out_file.txt')
    # parser.add_argument('csvF', type=str, help='path to input raw_file.csv')
    # parser.add_argument('featsF', type=str, help='path of output data/feats3_sentence.csv')
    # parser.add_argument('feats5F', type=str, help='path of output data/feats5_sentence.csv')
    # parser.add_argument('featsUserF', type=str, help='path of output data/feats3_user.csv')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    working_dir = '/Users/skymiao/PycharmProjects/audio-auditor-extension/'

    # # Load csv
    # args = get_arguments()
    # n_audio = args.n_audio
    # csv = pd.read_csv(csvF)
    # feats_file = args.featsF
    # feats5_file = args.feats5F
    # feats_user_file = args.featsUserF
    csv_F = "testing_auditor_100user/nonmember_test_clean_2_user.csv"
    # csv_F = "training_auditor_360shd_gru/nonmember_test_clean_1_shd.csv"
    feats_file = "data/gru_100_user/feats3_test-clean-2-user.csv"
    feats5_file = "data/gru_100_user/feats5_test-clean-2-user.csv"
    # feats_user3_file = "data/feats3U3_gru_train-clean-360-shd.csv"
    # feats_user2_file = "data/feats3U2_gru_train-clean-360-shd.csv"
    # feats_user4_file = "data/feats5U4_gru_train-clean-360-shd.csv"
    feats_user5_file = "data/gru_100_user/feats5U5A13_test-clean-2-user.csv"
    model_file = "glove_word2vec.txt"

    n_audio = 13
    path1 = feats5_file.split('/')
    path2 = path1[2].split('_', 1)
    feats5_Afile = path1[0] + '/' + path1[1] + '/' + path2[0] + 'U5A{}_'.format(n_audio) + path2[1]

    print ("=========================================")
    print ("Loading: csv_file = {}".format(csv_F))
    # print ("         feats_file = {}".format(feats_file))
    # print ("         feats5_file = {}".format(feats5_file))
    # print ("         feats_user3_file = {}".format(feats_user3_file))
    # print ("         feats_user2_file = {}".format(feats_user2_file))
    # print ("         feats_user5_file = {}".format(feats_user5_file))
    # print ("         feats_user4_file = {}".format(feats_user4_file))
    print ("         feats5_Afile = {}".format(feats5_Afile))
    csv = pd.read_csv(csv_F)

    # # Initial the list for processing original 4 feats into 3 feats except 'id'
    # # Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
    # #           ==> ['id', 'similarity', 'frame_length', 'speed']
    # print ("----------------------------------------")
    # model_type = model_file.split('.')[1]
    # if model_type == 'model':
    #     print ("Load pretrained Word2Vec model --- {}...".format(model_file))
    #     w2vModel = Word2Vec.load(working_dir + model_file)
    # elif model_type == 'txt':
    #     print ("Load pretrained Glove Word2Vec model --- {}...".format(model_file))
    #     w2vModel = KeyedVectors.load_word2vec_format(working_dir + model_file)
    # else:
    #     print ("ERROR: Load pretrained (Glove) Word2Vec model")
    #     sys.exit(1)
    # feats3 = feats3(csv, feats_file, w2vModel)

    # # Process the list of 3 feats into 5 feats except 'id'
    # # Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length'] = csv
    # # #         ==> ['id', 'similarity', 'frame_length', 'speed'] = feats3
    # #           ==> ['id', 'similarity', 'diff_miss', 'diff_add', 'frame_length', 'speed'] = feats5
    # feats3 = pd.read_csv(feats_file)
    # feats5 = feats5(csv, feats3, feats5_file)

    # # Statistically analyze the list for processing 3 feats towards each user where 'id' = user#-chapter#-sentence#
    # # Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
    # #           ==> ['user', 'similarity_statistics', 'frame_length_statistics', 'speed_statistics']
    # feats3 = pd.read_csv(feats_file)
    # feats3_user = feats_user3(feats3, feats_user3_file)
    # feats2_user = feats_user2(feats3, feats_user2_file)
    feats5 = pd.read_csv(feats5_file)
    feats5_user = feats_user5(csv, feats5, feats_user5_file, feats5_file)
    # feats4_user = feats_user4(csv, feats5, feats_user4_file, feats5_file)
    feats5_user = feats_user5_audio(csv, feats5, feats5_Afile, feats5_file, n_audio)


