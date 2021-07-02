# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:07:30 2019

@author: yuyu-
"""

import pandas as pd
import os
import re
import numpy as np
import pickle
import glob
from numpy import dot
from numpy.linalg import norm
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def get_vector(model, sentence):
    # convert to lowercase, ignore all special characters - keep only
    # alpha-numericals and spaces
    sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())

    #vectors = [model.wv[w] for w in word_tokenize(sentence) if w in model.wv]
    vectors = [model.wv[w] for w in sentence if w in model.wv]
    v = np.zeros(model.vector_size)

    if (len(vectors) > 0):
        v = (np.array([sum(x) for x in zip(*vectors)])) / v.size

    return v


def similarity(x, y):
    xv = get_vector(x)
    yv = get_vector(y)
    score = 0

    if xv.size > 0 and yv.size > 0:
        score = dot(xv, yv) / (norm(xv) * norm(yv))

    return score


def SavedPickle(path, file_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(file_to_save, handle)


def LoadPickleData(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


# Get the sentence vector for each auditor's audio (both predicted and true transcript.)
# Return [lines_predicted_vectors, lines_true_vectors]
def get_sentence_vector(lines_predicted_text, true_transcripts_text, w2vModel, pre_vector_pkl, true_vector_pkl):
    lines_predicted_vectors = []
    for item in lines_predicted_text:
        lines_predicted_vectors.append(get_vector(w2vModel, item))

    SavedPickle(working_dir + pre_vector_pkl, lines_predicted_vectors)

    lines_true_vectors = []
    for item in true_transcripts_text:
        lines_true_vectors.append(get_vector(w2vModel, item))

    SavedPickle(working_dir + true_vector_pkl, lines_true_vectors)

    return lines_predicted_vectors, lines_true_vectors


# Collect total samples or updated samples from Predicted logs
# Return [lines_predicted_text, lines_predicted_id, lines_predicted]
def lib_predicted(predicted_files):
    lines_predicted = []  # Predicted text
    lines_predicted_id = []
    lines_predicted_text = []

    for file_name in predicted_files:
        # with open(predicted_path + file_name, 'r') as file_to_read:
        with open(file_name, 'r') as file_to_read:
            lines = file_to_read.readlines()  # 整行读取数据
            for line in lines:
                start_of_line = line.split()[0]
                if start_of_line[0].isdigit():
                    lines_predicted.append(line.strip('\n').lower())  # Get the IDs and text content.
                    line_list = line.strip('\n').lower().split()
                    lines_predicted_id.append(line_list[0])
                    # lines_predicted_text.append(' '.join(line_list[1:]))
                    lines_predicted_text.append(line_list[1:])

    return lines_predicted_text, lines_predicted_id, lines_predicted


# Collect total samples or updated samples from true transcriptions
# Return [true_transcripts_text, true_transcripts_id, true_transcripts]
def lib_true(true_label_files):
    true_transcripts = []  # Predicted text
    true_transcripts_id = []
    true_transcripts_text = []

    for file_name in true_label_files:
        with open(true_label_path + file_name, 'r') as file_to_read:
            lines = file_to_read.readlines()  # 整行读取数据
            for line in lines:
                true_transcripts.append(line.strip('\n').lower())
                line_list = line.strip('\n').lower().split()
                true_transcripts_id.append(line_list[0])
                true_transcripts_text.append(line_list[1:])

    return true_transcripts_text, true_transcripts_id, true_transcripts


# Train the initialized Word2Vec model
# Return w2vModel and save the model as .model file
def initial_word2vec(working_dir, w2v_file, total_samples):
    print ("========================================")
    print ("Start training the Word2Vec model. Please wait.. ")
    # 2. Train a Vocabulary with Word2Vec -- using the function provided by gensim
    path = get_tmpfile(w2v_file)
    w2vModel = Word2Vec(total_samples, workers=8, window=2, min_count=1)
    print ("Trained with logs in testing_set_for_auditor/ and True_transcripts/ with {} records".format(
        len(total_samples)))
    print ("Model training completed!")
    print ("----------------------------------------")
    print ("The trained word2vec model: ")
    print (w2vModel)
    # w2vModel.wv.save_word2vec_format(working_dir + "predicted_and_true_text_w2v_model.txt", binary=False)
    w2vModel.save(working_dir + w2v_file)

    return w2vModel


# Update the initialized Word2Vec model
# Return w2vModel and save the model as new.model file
def update_word2vec(working_dir, w2v_model, update_samples, w2v_file2):
    print ("========================================")
    print ("Start updating the Word2Vec model = {}. Please wait.. ".format(w2v_model))
    w2vModel = Word2Vec.load(working_dir + w2v_model)
    w2vModel.build_vocab(update_samples, update=True)
    w2vModel.train(update_samples, total_examples=len(update_samples), epochs=w2vModel.epochs)
    print ("Updated with logs in training_set_for_auditor/ with {} records".format(len(update_samples)))
    print ("Model training completed!")
    print ("----------------------------------------")
    print ("The updated word2vec model as {}: ".format(w2v_file2))
    print (w2vModel)
    # w2vModel.wv.save_word2vec_format(working_dir + "predicted_and_true_text_w2v_model.txt", binary=False)
    w2vModel.save(working_dir + w2v_file2)

    return w2vModel


if __name__ == '__main__':

    # # Collect total samples or updated samples from Predicted logs
    # predicted_path = working_dir + 'train-clean-100_360_predicted' + os.sep
    # predicted_files= os.listdir(predicted_path)
    # predicted_path = working_dir + 'testing_set_for_auditor' + os.sep
    predicted_path = 'training_set_for_auditor' + os.sep
    predicted_files= glob.glob(predicted_path + '*/*.log')

    lines_predicted_text, lines_predicted_id, lines_predicted = lib_predicted(predicted_files)

    # Collect total samples or updated samples from true transcriptions
    # Comment for update model
    true_label_path = 'True_transcripts' + os.sep
    true_label_files = os.listdir(true_label_path)

    true_transcripts_text, true_transcripts_id, true_transcripts = lib_true(true_label_files)

    # Initialize model / Update model
    total_samples = lines_predicted_text + true_transcripts_text
    # update_samples = lines_predicted_text

    # Train the initialized Word2Vec model
    w2v_file = "word2vec_libri.model"
    w2vModel = initial_word2vec(working_dir, w2v_file, total_samples)

    # # Update the pretrained model with another total_samples
    # w2v_model = "word2vec_libri_2.model"
    # w2v_file2 = "word2vec_libri_3.model"
    # w2vModel = update_word2vec(working_dir, w2v_model, update_samples, w2v_file2)


    # # Get the sentence vector for each auditor's audio (both predicted and true transcript.)
    # pre_vector_pkl = 'lines_predicted_vectors.pkl'
    # true_vector_pkl = 'lines_true_vectors.pkl'
    # lines_predicted_vectors, lines_true_vectors = get_sentence_vector(lines_predicted_text, true_transcripts_text,
    #                                                                   w2vModel, pre_vector_pkl, true_vector_pkl)
