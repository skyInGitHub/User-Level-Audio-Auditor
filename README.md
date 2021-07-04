# user-level-audio-auditor (Transcriptions-Only)
Paper: [The Audio Auditor: User-Level Membership Inference in Internet of Things Voice Services](https://arxiv.org/abs/1905.07082)

Published: PoPETS 2019

## Table of Contents
* [Methodology](#methodology)
* [Data Prepare](#data-prepare)
* [Data Preprocess](#data-preprocess-feature-extraction)
* [User-level Audio Auditor Model](#user-level-audio-auditor-model)
* [Result Analysis](#result-analysis)

----
## Methodology

Transcription-only black-box access to ASR model:
* Input: audio & its true transcription
* Output: its predicted transcription

User-level Membership Inference Attack:
Querying with a user’s data, if this user has any data within target model’s training set, even if the query data are not members of the training set, this user is the user-level member of this training set.

Fig. 2 depicts a workflow of our audio auditor auditing an ASR model. Generally, there are two processes, i.e., training and auditing. The former process is to build a binary classifier as a user-level membership auditor *A_{audit}* using a supervised learning algorithm. The latter uses this auditor to audit an ASR model *F_{tar}* by querying a few audios spoken by one user u. In Section 4.4, we show that only a small number of audios per user can determine whether *u ∈ U_{tar}* or *u ∈/ U_{tar}*. Furthermore, a small number of users used to train the auditor is sufficient to provide a satisfying result.

<img width="853" alt="methodology" src="https://user-images.githubusercontent.com/13388819/124377778-ed815200-dcec-11eb-9cda-d13eb265dc08.png">


----
## Data Prepare
Each matrix obtains 4 columns {id, transcript_txt, frame_length, txt, txt_length}.

For record id=777-126732-0046, under the folder ./testing_set_for_auditor,
extract elements from /decode_dev_clean_2out_dnn2/decode.1.log and dev-clean-2-true-txt.txt

{777-126732-0046, "IN ANY CASE HE HAD NOT THE TIME", 223, "IN ANY CASE HE HAD NOT THE TIME", 31}

1. log to txt. 

_decodelog2txt.sh_: 
* input: dataset = $1 = testing_set_for_auditor/decode_test_clean_2_user_out_dnn2; label = $2 = nonmember_test_clean_2_user
* output: txt_f = testing_set_for_auditor/nonmember_test_clean_2_user.txt
* Reduce irrelevant information from the raw transcription results.

```bash
$ ./decodelog2txt.sh testing_set_for_auditor/decode_train_clean_100_user_out_dnn2 member_train_clean_100_user 
>> out_log/decodelog2txt_member_train_clean_100_user.txt 2>&1 && echo 's' || echo 'e'
$ cp out_log/decodelog2txt_member_train_clean_100_user.txt testing_set_for_auditor/
```

2. txt to csv.

_txt2csv.py_: 
* input: txt_in = "testing_set_for_auditor/nonmember_test_clean_2_user.txt"; 
         true_in = "testing_set_for_auditor/test-clean-2-user-true-txt.txt"
* output: csv_file = "data/nonmember_test-clean-2-user.csv"
* Convert txt_in (transcription results) to a matrix focusing on sentence id.
* Extract 4 features (trans_txt, frame_length, true_txt, true_txt_length) for each sentence id.
* Save as .csv focusing on sentence id. header = ['id', 'predicted_txt', 'true_txt', 'true_txt_length', 'frame_length'].

```bash
$ python ./txt2csv.py 
```

## Data Preprocess (feature extraction)
Transfer **sentence-id-record** ('**id**') data to **user-id-record** ('**user**') data.
Mainly process 2 string-type features --- 'predicted_txt' and 'true_txt' --- into int-type features as similarity score. 
The other 2 int-type features including previous processed 2 int-type features are analyzed statistically.

1. Word2Vec Model Training

_word_embedding.py_:
* input: predicted_path = testing_set_for_auditor/*/*.log; 
         true_label_path = True_transcripts/*.txt
* output: w2vModel = word2vec_libri.model
* Train a Word2Vec with 2 kinds of Vocabularies (logs and true_txt files) --> save as .model
* Update the pretrained model (word2vec_*.model) with another total_samples.

```bash
$ python ./word_embedding.py
```

2. Word2Vec Model Update

New log files found:
* Repeat ## Data Preprocess 1. Word2Vec Model Training
* Repeat 

```bash
$ python ./word_embedding.py
```

3. Similarity Score Between 'predicted_txt' and 'true_txt'

_feature_sentence.py_:
* input: csv = data/member_dev-clean-2.csv
* output: feats_file = data/member_feats3_dev-clean-2.csv
* Load pretrained Word2Vec model (word2vec_*.model)
* Initial the list for processing original 4 feats into 3 feats except 'id'
  Convert the 2nd and 3rd columns (string-type features) into word vectors --- 1 word 1 vector and 1 similarity score.
  Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
            ==> ['id', 'similarity', 'frame_length', 'speed']
* Save initial features as .csv focusing on sentence id

```bash
$ python ./feature_sentence.py
```

4. Similarity Statistic for Each User

_feature_speaker.py_:
* input: feats_csv = data/member_feats3_dev-clean-2.csv
* output: feats_user_file = data/member_feats3_user_dev-clean-2.csv
* Statistically analyze the list for processing 3 feats towards each user where 'id' = user#-chapter#-sentence#
  Specifically, ['id', 'predicted_text', 'true_text', 'true_text_length', 'frame_length']
             ==> ['user', 'similarity_statistics', 'frame_length_statistics', 'speed_statistics']
* Save processed features as .csv focusing on user(speaker) id

```bash
$ python ./feature_speaker.py
```

## User-level Audio Auditor Model 

```bash
$ python ./audit_speaker.py
```


