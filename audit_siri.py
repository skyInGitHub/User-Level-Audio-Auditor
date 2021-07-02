import pandas as pd
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def confu_result(y_test, y_pred, result_txt):

    confu_matrix = confusion_matrix(y_test, y_pred)
    # pd.DataFrame(confu_matrix).to_csv(confu_csv)

    result_report = classification_report(y_test, y_pred, output_dict=True)
    result_report = pd.DataFrame(result_report).transpose()

    accuracy = accuracy_score(y_test, y_pred)
    result_report['Accuracy'] = accuracy
    # result_report.to_csv(result_txt)

    return confu_matrix, result_report, accuracy


def auditor_test(test_f, result_txt, filename, repeat_time, result_roc):
    # Load labelled datasets for different sets
    test_initial = pd.read_csv(test_f)
    print ("=======================================")
    print ("Loading: test_set = {}".format(test_f))
    print ("         result_txt = {}".format(result_txt))

    # Randomly sample 10 times & Get the average result
    index = ['macro avg', 'member', 'micro avg', 'nonmember', 'weighted avg']
    feature = ['f1-score', 'precision', 'recall', 'support', 'Accuracy']
    report_avg = pd.DataFrame(0, index=index, columns=feature)
    report_max = pd.DataFrame()
    max_acc = 0
    lr_auc_avg = 0

    # print ("------------------------------------")
    print ("Load the auditor model...")
    classifier = pickle.load(open(filename, 'rb'))

    for i in range(repeat_time):
        # Train the auditor model
        # print ("------------------------iterate {}---------------------".format(i))
        # print ("Prepare training and testing set...")
        test_set = test_initial.sample(n=100, replace=True)
        # test_set = test_initial
        test_set = test_set.drop('user', axis=1)
        X_test = test_set.drop('class', axis=1)
        y_test = test_set['class']
        ns_probs = [0 for _ in range(len(y_test))]

        y_pred = classifier.predict(X_test)
        [confu_matrix, result_report, accuracy] = confu_result(y_test, y_pred, result_txt)
        report_avg += result_report

        lr_probs = classifier.predict_proba(X_test)
        lr_probs = lr_probs[:, 1]
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        # # summarize scores
        # print('No Skill: ROC AUC=%.3f' % (ns_auc))
        # print('Logistic: ROC AUC=%.3f\n' % (lr_auc))
        lr_auc_avg += lr_auc

        y_test_bin = y_test.reset_index(drop=True)
        for j in range(len(y_test_bin)):
            if y_test_bin.loc[j] == 'member':
                y_test_bin.loc[j] = 1
            else:
                y_test_bin.loc[j] = 0
        ns_fpr, ns_tpr, _ = roc_curve(y_test_bin, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test_bin, lr_probs)
        # roc_cur['ns_fpr'] = ns_fpr
        # roc_cur['ns_tpr'] = ns_tpr
        roc_cur = pd.DataFrame()
        roc_cur['lr_fpr'] = lr_fpr
        roc_cur['lr_tpr'] = lr_tpr
        if i == 0:
            roc_cur_avg = roc_cur
        else:
            roc_cur_avg = roc_cur + roc_cur_avg

        # # plot the roc curve for the model
        # pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        # plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # # axis labels
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # # show the legend
        # plt.legend()
        # # show the plot
        # plt.show()
        # plt.savefig(result_roc)

        if accuracy > max_acc:
            max_acc = accuracy
            report_max = result_report
        # print("The Accuracy is: {}, while max is {}.".format(accuracy, max_acc))

    report_avg = report_avg / repeat_time
    report_avg.to_csv(result_txt)
    roc_cur_avg = roc_cur_avg / repeat_time
    roc_cur_avg.to_csv(result_roc)

    lr_auc_avg = lr_auc_avg / repeat_time

    # # report_max.to_csv(result_txt.split('.')[0] + '_max.csv')
    # print("------------------------------------------------------")
    # print("The classification report is located at: {}\n {}".format(result_txt, report_avg))
    print("The average roc auc for this model is: {}\n".format(lr_auc_avg))
    # print("The ROC Curve report is located at: {}\n {}".format(result_roc, roc_cur_avg))

    # return confu_matrix, result_report, accuracy, roc_cur, lr_auc
    return report_avg, roc_cur_avg, lr_auc_avg


def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('n_audio', type=int, help='# of querying audios')
    parser.add_argument('all', type=int, help='n of sample for mem + nonmem in training.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    # # Auditor Testing
    # args = get_arguments()
    # n_audio = args.n_audio
    # all = args.all
    # n_audio = 5
    all = 1000      # 1000/2000
    test_f = "data/result_siri/test_siri_memU6.csv" # modified for all mem + nonmem
    # result_txt = "data/result_siri/u55A5_RF_A5_siri/u55A5_RF_A5_{}.csv".format(all)

    # Randomly sample 10 times & Get the average result
    result_txt_avg = "data/result_siri/u55A5_RF_A5_siri_U6/avg_u55A5_RF_A5_U6_{}.csv".format(all)
    result_roc_avg = "data/result_siri/u55A5_RF_A5_siri_U6_roc/avg_u55A5_RF_A5_U6_{}.csv".format(all)
    index = ['macro avg', 'member', 'micro avg', 'nonmember', 'weighted avg']
    feature = ['f1-score', 'precision', 'recall', 'support', 'Accuracy']
    report_avg = pd.DataFrame(0, index=index, columns=feature)
    report_max = pd.DataFrame()
    max_acc = 0
    lr_auc_avg = 0

    times = 100
    for i in range(times):
        result_txt = "data/result_siri/u55A5_RF_A5_siri_U6/u55A5_RF_A5_U6_{}_{}.csv".format(all, i)
        result_roc = "data/result_siri/u55A5_RF_A5_siri_U6_roc/u55A5_RF_A5_U6_{}_{}.csv".format(all, i)
        filename ='data/result_siri/100_360gru/Auditor_model_A5_{}_{}.sav'.format(all, i)

        repeat_time = 1

        # # Load labelled datasets for different sets
        # test_initial = pd.read_csv(test_f)
        # print ("=======================================")
        # print ("Loading: test_set = {}".format(test_f))
        # print ("         result_txt = {}".format(result_txt))

        [result_report, roc_cur, lr_auc] = auditor_test(test_f, result_txt, filename, repeat_time, result_roc)
        report_avg += result_report
        lr_auc_avg += lr_auc
        # if i == 0:
        #     roc_cur_avg = roc_cur
        # else:
        #     roc_cur_avg = roc_cur + roc_cur_avg


        # if accuracy > max_acc:
        #     max_acc = accuracy
        #     report_max = result_report
        # print("The Accuracy is: {}, while max is {}.".format(accuracy, max_acc))

    report_avg = report_avg / times
    report_avg.to_csv(result_txt_avg)
    # roc_cur_avg = roc_cur_avg / times
    # roc_cur_avg.to_csv(result_roc_avg)
    lr_auc_avg = lr_auc_avg / times
    # report_max.to_csv(result_txt.split('.')[0] + '_max.csv')
    print("------------------------------------------------------")
    print("The classification report is located at: {}\n {}".format(result_txt_avg, report_avg))
    print("The average roc auc is: {}\n".format(lr_auc_avg))
    # print("The ROC Curve report is located at: {}\n {}".format(result_roc_avg, roc_cur_avg))


