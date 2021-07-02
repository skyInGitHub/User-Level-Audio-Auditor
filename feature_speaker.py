import pandas as pd
import argparse


def load_unlabelled_train(train_trainl_csv, train_testl_csv, train_f):
    # Load datasets
    print ("=======================================")
    print ("Loading: train_trainL_csv = {}".format(train_trainl_csv))
    print ("         train_testL_csv = {}".format(train_testl_csv))
    train_trainL_set = pd.read_csv(train_trainl_csv)
    train_testL_set = pd.read_csv(train_testl_csv)

    # Set up ground truth for training set of our auditor
    print ("Set up ground truth for each user...")
    train_mem = 0
    train_nonmem = 0
    Labels_train = []
    for n in range(len(train_trainL_set)):
        Labels_train.append("member")
        train_mem += 1
    train_trainL_set['class'] = Labels_train

    Labels_test = []
    for nrow in range(len(train_testL_set)):
        if train_testL_set.at[nrow, 'user'] in train_trainL_set['user']:
            Labels_test.append("member")
            train_mem += 1
        else:
            Labels_test.append("nonmember")
            train_nonmem += 1
    train_testL_set['class'] = Labels_test

    train_set = train_trainL_set
    train_set = train_set.append(train_testL_set)
    pd.DataFrame(train_set).to_csv(train_f, index=None)
    print (">>Training set for auditor: {} spk = {} train-clean-360-shd + "
           "{} test-clean-1-shd.".format(len(train_set), len(train_trainL_set), len(train_testL_set)))
    print ("  Training set for auditor has {} mem and {} nonmem.".format(train_mem, train_nonmem))
    print ("  Training set saves as {}".format(train_f))

    # Extract member in shadow model's testing set
    train_testL_mem = train_testL_set.loc[train_testL_set['class'] == 'member']
    print ("  Training set include {} user'member in testing restult".format(len(train_testL_mem)))

    # Training set for auditor (Exclude member in shadow model's testing set)
    if len(train_testL_mem) != 0:
        nonmem_set = train_testL_set.loc[train_testL_set['class'] == 'nonmember']
        train_testL_new = train_trainL_set
        train_testL_new = train_testL_new.append(nonmem_set)
        path = train_f.split('.')[0] + '_inout.csv'
        pd.DataFrame(train_testL_new).to_csv(path, index=None)
        print("Training set exclude user'member has {} mem + {} nonmem.".format(len(train_trainL_set), len(nonmem_set)))

    return train_set, train_trainL_set, train_testL_set


def load_unlabelled_test(test_trainl_csv, test_testl_csv, test_f):
    # Load datasets
    print ("=======================================")
    print ("Loading: test_trainL_csv = {}".format(test_trainl_csv))
    print ("         test_testL_csv = {}".format(test_testl_csv))
    test_trainL_set = pd.read_csv(test_trainl_csv)
    test_testL_set = pd.read_csv(test_testl_csv)

    # Set up ground truth for testing set of our auditor
    print ("Set up ground truth for each user...")
    test_mem = 0
    test_nonmem = 0
    Labels_train = []
    for n in range(len(test_trainL_set)):
        Labels_train.append("member")
        test_mem += 1
    test_trainL_set['class'] = Labels_train

    Labels_test = []
    for nrow in range(len(test_testL_set)):
        if test_testL_set.at[nrow, 'user'] in test_trainL_set['user']:
            Labels_test.append("member")
            test_mem += 1
        else:
            Labels_test.append("nonmember")
            test_nonmem += 1
    test_testL_set['class'] = Labels_test
    path = test_f.split('.')[0] + '_allout.csv'
    pd.DataFrame(test_testL_set).to_csv(path, index=None)

    test_set = test_trainL_set
    test_set = test_set.append(test_testL_set)
    pd.DataFrame(test_set).to_csv(test_f, index=None)
    print (">>Testing set for auditor: {} spk = {} train-clean-100-user + "
           "{} test-clean-2-user.".format(len(test_set), len(test_trainL_set), len(test_testL_set)))
    print ("  Testing set for auditor has {} mem and {} nonmem.".format(test_mem, test_nonmem))
    print ("  Testing set saves as {}".format(test_f))

    # Extract member in target model's testing set
    test_testL_mem = test_testL_set.loc[test_testL_set['class'] == 'member']
    path = test_f.split('.')[0] + '_memout.csv'
    pd.DataFrame(test_testL_mem).to_csv(path, index=None)
    print ("  Testing set include {} usermember in target model's testing set".format(len(test_testL_mem)))

    # Extract member in target model's training set
    path = test_f.split('.')[0] + '_memin.csv'
    pd.DataFrame(test_trainL_set).to_csv(path, index=None)
    print ("  Testing set include {} usermember in target model's training set".format(len(test_trainL_set)))

    # Training set for auditor (Exclude member in target model's testing set)
    if len(test_testL_mem) != 0:
        nonmem_set = test_testL_set.loc[test_testL_set['class'] == 'nonmember']
        path = test_f.split('.')[0] + '_memnon.csv'
        pd.DataFrame(nonmem_set).to_csv(path, index=None)
        test_testL_new = test_trainL_set
        test_testL_new = test_testL_new.append(nonmem_set)
        path = test_f.split('.')[0] + '_inout.csv'
        pd.DataFrame(test_testL_new).to_csv(path, index=None)
        print("Testing set exclude user'member has {} mem + {} nonmem.".format(len(test_trainL_set), len(nonmem_set)))

    return test_set, test_trainL_set, test_testL_set


def load_unlabelled(train_trainl_csv, train_testl_csv, test_trainl_csv, test_testl_csv, train_f, test_f):
    # Load datasets
    print ("=======================================")
    print ("Loading: train_trainL_csv = {}".format(train_trainl_csv))
    print ("         train_testL_csv = {}".format(train_testl_csv))
    print ("         test_trainL_csv = {}".format(test_trainl_csv))
    print ("         test_testL_csv = {}".format(test_testl_csv))
    train_trainL_set = pd.read_csv(train_trainl_csv)
    train_testL_set = pd.read_csv(train_testl_csv)
    test_trainL_set = pd.read_csv(test_trainl_csv)
    test_testL_set = pd.read_csv(test_testl_csv)

    # Set up ground truth for training and testing set of our auditor
    print ("Set up ground truth for each user...")
    # Ground truth for training set
    train_mem = 0
    train_nonmem = 0
    Labels_train = []
    for n in range(len(train_trainL_set)):
        Labels_train.append("member")
        train_mem += 1
    train_trainL_set['class'] = Labels_train

    Labels_test = []
    for nrow in range(len(train_testL_set)):
        if train_testL_set.at[nrow, 'user'] in train_trainL_set['user']:
            Labels_test.append("member")
            train_mem += 1
        else:
            Labels_test.append("nonmember")
            train_nonmem += 1
    train_testL_set['class'] = Labels_test

    train_set = train_trainL_set
    train_set = train_set.append(train_testL_set)
    pd.DataFrame(train_set).to_csv(train_f, index=None)
    print (">>Training set for auditor: {} spk = {} train-clean-360-shd + "
           "{} test-clean-1-shd.".format(len(train_set), len(train_trainL_set), len(train_testL_set)))
    print ("  Training set for auditor has {} mem and {} nonmem.".format(train_mem, train_nonmem))
    print ("  Training set saves as {}".format(train_f))

    # Ground truth for testing set
    test_mem = 0
    test_nonmem = 0
    Labels_train = []
    for n in range(len(test_trainL_set)):
        Labels_train.append("member")
        test_mem += 1
    test_trainL_set['class'] = Labels_train

    Labels_test = []
    for nrow in range(len(test_testL_set)):
        if test_testL_set.at[nrow, 'user'] in test_trainL_set['user']:
            Labels_test.append("member")
            test_mem += 1
        else:
            Labels_test.append("nonmember")
            test_nonmem += 1
    test_testL_set['class'] = Labels_test

    test_set = test_trainL_set
    test_set = test_set.append(test_testL_set)
    pd.DataFrame(test_set).to_csv(test_f, index=None)
    print (">>Testing set for auditor: {} spk = {} train-clean-100-user + "
           "{} test-clean-2-user.".format(len(test_set), len(test_trainL_set),len(test_testL_set)))
    print ("  Testing set for auditor has {} mem and {} nonmem.".format(test_mem, test_nonmem))
    print ("  Testing set saves as {}".format(test_f))

    return train_set, test_set


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_audio', type=int, help='# of querying audios')
    # parser.add_argument('n_sample', type=int, help='the amount number of random users/features')
    # parser.add_argument('n_time', type=int, help='nth time for average result.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    # args = get_arguments()
    # n_audio = args.n_audio

    # # Set up speaker level datasets
    # train_trainL_csv = "data/gru_360_shd/feats5U5A1_gru_train-clean-360-shd.csv"
    # train_testL_csv = "data/gru_360_shd/feats5U5A1_gru_test-clean-1-shd.csv"
    test_trainL_csv = "data/lstm_100_user/feats5U5A13_train-clean-100-user.csv"
    test_testL_csv = "data/lstm_100_user/feats5U5A13_test-clean-2-user.csv"

    # train_f = "data/train/train_u55_360gru.csv"
    test_f = "data/test2/test_u55A13_100.csv"

    # # Load unlabelled datasets & Set up ground truth (user-level)
    # [train_set, test_set] = load_unlabelled(train_trainL_csv, train_testL_csv, test_trainL_csv, test_testL_csv, train_f, test_f)
    # [train_set, train_trainL_set, train_testL_set] = load_unlabelled_train(train_trainL_csv, train_testL_csv, train_f)
    [test_set, test_trainL_set, test_testL_set] = load_unlabelled_test(test_trainL_csv, test_testL_csv, test_f)

    # label = "nonmember"
    # load_unlabelled_single(test_testL_csv, test_f, label)



