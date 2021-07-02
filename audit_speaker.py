import os
import sys
import glob
import pandas as pd
import argparse
import numpy as np
from random import sample
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import pickle


def label_encoder(train_data):
    le = preprocessing.LabelEncoder()

    for column_name in train_data.columns:
        if train_data[column_name].dtype == object:
            train_data[column_name] = le.fit_transform(train_data[column_name])
        else:
            pass

    return train_data


def list_sum(train_data):
    diff_miss = train_data['diff_miss'].tolist()
    diff_add = train_data['diff_add'].tolist()

    for nrow in range(len(train_data)):
        miss = list(diff_miss[nrow].replace('[', '').replace(']', '').split(', '))
        miss = [int(x) for x in miss]
        train_data.at[nrow, 'diff_miss'] = sum(miss)

        add = list(diff_add[nrow].replace('[', '').replace(']', '').split(', '))
        add = [int(x) for x in add]
        train_data.at[nrow, 'diff_add'] = sum(add)

    return train_data


def random_train(X_train, y_train, n_sample):

    train_len = len(X_train)
    train_indices = sample(range(train_len), n_sample)

    X_train = X_train.iloc[train_indices]
    y_train = y_train.iloc[train_indices]

    return X_train, y_train


def random_test_feature(X_test, n_sentence):

    test_fea = [0, 3, 6, 9, 12, 15, 18, 21]
    indices = sample(test_fea, n_sentence)
    test_indices = []
    for i in indices:
        test_indices.append(int(i))
        test_indices.append(int(i+1))
        test_indices.append(int(i+2))

    # X_test = X_test.iloc[test_indices]

    # test_not = []
    for i in range(24):
        if i not in test_indices:
            # test_not.append(i)
            X_test.iloc[:, i] = -1

    return X_test


def avg_results(chdir):
    os.chdir(chdir)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    avg_pre = 0
    avg_recall = 0
    avg_f1 = 0
    avg_sup = 0

    for f in all_filenames:
        result_csv = pd.read_csv(f)


def confu_result(y_test, y_pred, result_txt):

    confu_matrix = confusion_matrix(y_test, y_pred)
    # pd.DataFrame(confu_matrix).to_csv(confu_csv)

    result_report = classification_report(y_test, y_pred, output_dict=True)
    result_report = pd.DataFrame(result_report).transpose()

    accuracy = accuracy_score(y_test, y_pred)
    result_report['Accuracy'] = accuracy
    # result_report.to_csv(result_txt)

    return confu_matrix, result_report, accuracy


def random_member(train_set, i, sample):
    train_new = pd.DataFrame(columns=train_set.columns.values.tolist())

    # Search each row/users in training set
    train_mem = train_set.loc[train_set['class'] == 'member']
    train_nonmem = train_set.loc[train_set['class'] == 'nonmember']
    n_mem = len(train_mem)
    n_nonmem = len(train_nonmem)
    if i == 0:
        print ("Process training set: {} mem_user = {} nonmem_user".format(n_mem, n_nonmem))

    # Reduce the same user id in the longer mem/nonmem set
    if n_mem > n_nonmem:
        user_nonmem = train_nonmem['user'].to_list()
        index_mem = train_mem[train_mem['user'].isin(user_nonmem)].index
        train_mem = train_mem.drop(index_mem)
        if i == 0:
            print ("Reduce common user: {} mem - {} index_mem = {} train_mem".format(n_mem, len(index_mem), len(train_mem)))
        n_mem = len(train_mem)
    else:
        user_mem = train_mem['user'].to_list()
        index_nonmem = train_nonmem[train_nonmem['user'].isin(user_mem)].index
        train_nonmem = train_nonmem.drop(index_nonmem)
        if i == 0:
            print ("Reduce common user: {} nonmem - {} index_nonmem = {} train_nonmem".format(n_nonmem, len(index_nonmem),
                                                                                           len(train_nonmem)))
        n_nonmem = len(train_nonmem)

    # Random chose {sample} mem/nonmem separately
    if sample <= n_nonmem and sample <= n_mem:
        train_new = train_new.append(train_mem.sample(n=sample), ignore_index=True)
        train_new = train_new.append(train_nonmem.sample(n=sample), ignore_index=True)

    elif sample <= n_nonmem:
        train_new = train_new.append(train_mem, ignore_index=True)
        rest = sample - n_mem
        train_new = train_new.append(train_mem.sample(n=rest, replace=True), ignore_index=True)

        train_new = train_new.append(train_nonmem.sample(n=sample), ignore_index=True)

    elif sample <= n_mem:
        train_new = train_new.append(train_mem.sample(n=sample), ignore_index=True)

        train_new = train_new.append(train_nonmem, ignore_index=True)
        rest = sample - n_nonmem
        train_new = train_new.append(train_nonmem.sample(n=rest, replace=True), ignore_index=True)

    else:
        train_new = train_new.append(train_mem, ignore_index=True)
        rest = sample - n_mem
        train_new = train_new.append(train_mem.sample(n=rest, replace=True), ignore_index=True)

        train_new = train_new.append(train_nonmem, ignore_index=True)
        rest = sample - n_nonmem
        train_new = train_new.append(train_nonmem.sample(n=rest, replace=True), ignore_index=True)

    # Check sample right or not
    if len(train_new) != 2*sample:
        print ("Error: len(train_new)={} != 2 * sample={}".format(len(train_new), sample))
        exit(0)

    if i == 0:
        print ("Randomly chose mem/nonmem users: let mem = nonmem".format(sample))
        print ("===> Complete train_new = {} = 2 * {} mem".format(len(train_new), sample))
        # print ("Randomly chose mem/nonmem users: let mem = nonmem = {}".format(min(n_nonmem, n_mem)))
        # print ("===> Complete train_new = {} = 2 * {} min(mem, nonmen)".format(len(train_new), min(n_nonmem, n_mem)))

    return train_new


def random_member_r(train_set,  i, sample):
    train_new = pd.DataFrame(columns=train_set.columns.values.tolist())

    # Count member # and nonmember # in column of 'class'
    train_nonmem = train_set.loc[train_set['class'] == 'nonmember']
    train_mem = train_set.loc[train_set['class'] == 'member']
    n_nonmem = len(train_nonmem)
    n_mem = len(train_mem)
    if i == 0:
        print ("Process training set: {} mem_user = {} nonmem_user".format(n_mem, n_nonmem))

    # Random chose {sample} mem/nonmem separately
    if sample <= n_nonmem and sample <= n_mem:
        train_new = train_new.append(train_mem.sample(n=sample), ignore_index=True)
        train_new = train_new.append(train_nonmem.sample(n=sample), ignore_index=True)

    elif sample <= n_nonmem:
        train_new = train_new.append(train_mem, ignore_index=True)
        rest = sample - n_mem
        train_new = train_new.append(train_mem.sample(n=rest, replace=True), ignore_index=True)

        train_new = train_new.append(train_nonmem.sample(n=sample), ignore_index=True)

    elif sample <= n_mem:
        train_new = train_new.append(train_mem.sample(n=sample), ignore_index=True)

        train_new = train_new.append(train_nonmem, ignore_index=True)
        rest = sample - n_nonmem
        train_new = train_new.append(train_nonmem.sample(n=rest, replace=True), ignore_index=True)

    else:
        train_new = train_new.append(train_mem, ignore_index=True)
        rest = sample - n_mem
        train_new = train_new.append(train_mem.sample(n=rest, replace=True), ignore_index=True)

        train_new = train_new.append(train_nonmem, ignore_index=True)
        rest = sample - n_nonmem
        train_new = train_new.append(train_nonmem.sample(n=rest, replace=True), ignore_index=True)

    # Check sample right or not
    if len(train_new) != 2*sample:
        print ("Error: len(train_new)={} != 2 * sample={}".format(len(train_new), sample))
        exit(0)

    if i == 0:
        print ("Randomly chose mem/nonmem users: let mem = nonmem".format(sample))
        print ("===> Complete train_new = {} = 2 * {} mem".format(len(train_new), sample))

    return train_new


def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('n_audio', type=int, help='# of querying audios')
    parser.add_argument('all', type=int, help='n of sample for mem + nonmem in training.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    # # Set up speaker level datasets
    # args = get_arguments()
    # n_audio = args.n_audio
    # n_audio = 13
    # all = args.all
    all = 1000
    sample = all / 2
    train_f = "data/train/train_u55A5_360gru_inout.csv"
    # test_f = "data/test/test_u55A{}_100.csv".format(n_audio)
    # result_txt = "data/result_100_360gru/u55A5_RF_inoutA{}/u55A5_RF_A{}_{}.csv".format(n_audio, n_audio, all)
    test_f = "data/test/test_u55_100.csv"
    result_txt = "data/result_100_360gru/u55A5_RF_inout/u55_RF_inout_{}.csv".format(all)   # NB, SVM, RF, KNN3, DT

    # Load labelled datasets for different sets
    train_initial = pd.read_csv(train_f)
    test_initial = pd.read_csv(test_f)
    print ("=======================================")
    print ("Loading: train_set = {}".format(train_f))
    print ("         test_set = {}".format(test_f))
    print ("         result_txt = {}".format(result_txt))

    # Randomly sample 10 times & Get the average result
    index = ['macro avg', 'member', 'micro avg', 'nonmember', 'weighted avg']
    feature = ['f1-score', 'precision', 'recall', 'support', 'Accuracy']
    report_avg = pd.DataFrame(0, index=index, columns=feature)
    report_max = pd.DataFrame()
    max_acc = 0
    repeat_time = 100

    for i in range(repeat_time):
        # Train the auditor model
        # print ("------------------------iterate {}---------------------".format(i))
        # print ("Prepare training and testing set...")
        # test_set = test_initial.sample(n=100, replace=True)
        test_set = test_initial
        train_set = random_member(train_initial, i, sample)
        train_set = train_set.drop('user', axis=1)
        test_set = test_set.drop('user', axis=1)

        X_train = train_set.drop('class', axis=1)
        y_train = train_set['class']
        X_test = test_set.drop('class', axis=1)
        y_test = test_set['class']

        # # print ("------------------------------------")
        # print ("Train the auditor model...")
        classifier = RandomForestClassifier(n_estimators=100)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        [confu_matrix, result_report, accuracy] = confu_result(y_test, y_pred, result_txt)
        report_avg += result_report
        result_report.to_csv(result_txt)

        # filename = 'data/result_100_360gru/u55A5_RF_inoutA{}/Auditor_model_A{}_{}_{}.sav'.format(n_audio, n_audio, all, i)
        # pickle.dump(classifier, open(filename, 'wb'))
        # print("The classification report of Auditor_model_A{}_{}_{}.sav is:\n {}".format(n_audio, all, i, result_report))

        if accuracy > max_acc:
            max_acc = accuracy
            report_max = result_report
            # # save the model to disk
            # filename = 'data/result_100_360gru/u55A5_RF_A{}/Auditor_model_A{}_{}.sav'.format(n_audio, n_audio, all)
            # pickle.dump(classifier, open(filename, 'wb'))

    report_avg = report_avg / repeat_time
    report_avg.to_csv(result_txt)
    report_max.to_csv(result_txt.split('.')[0]+'_max.csv')
    print("------------------------------------------------------")
    print("The classification report of average is located at: {}\n {}".format(result_txt, report_avg))
    print("The classification report of max is:\n {}".format(report_max))
    print ("END")
