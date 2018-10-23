#!usr/bin/python
# -*- coding:utf-8 -*-

from gensim.models import word2vec
import jieba
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import codecs
import re
import time


def text_word_frequency(data_file, wv_model, fstop_list):
    '''
    求出所有训练样本的特征的词频
    :param data:
    :return:
    '''
    all_feature_freq = []
    data_label = []
    pattern = re.compile(u"[\d\s+\.\!\/_,$%^*()?;；。：、:-【】+\"\']+|[+——！，;:。：？、~@#￥%……&* -]+|[()（）～]+")

    for num in range(len(data_file)):
        pre_data = re.sub(pattern, '', data_file.loc[num, 'content'])
        data_label.append(int(data_file.loc[num, 'label']))
        data_list = jieba.lcut(pre_data)
        flag = 0
        for word in data_list:
            if word not in fstop_list and len(word) >= 2:
                flag += 1
                if flag == 1:
                    word_vec = np.array(wv_model.wv[word])
                else:
                    word_vec += np.array(wv_model.wv[word])
            else:
                continue
        sen_vec = word_vec/flag
        all_feature_freq.append(sen_vec)
    return all_feature_freq, data_label


if __name__ == '__main__':
    start_time = time.clock()
    wv_model = word2vec.Word2Vec.load('word2vec_model.txt')

    stopwords_path = 'stoplist.txt'  # 停用词词表
    f_stop = codecs.open(stopwords_path, "rb", 'utf-8')
    fstop_list = f_stop.read()

    pattern = re.compile(u"[\d\s+\.\!\/_,$%^*()?;；。：、:-【】+\"\']+|[+——！，;:。：？、~@#￥%……&* -]+|[()（）～]+")

    file_path = 'origin_dataset.csv'
    data_file = pd.read_csv(file_path, encoding='gb18030')

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []

    skf = StratifiedKFold(n_splits=5)          # 5折交叉验证
    for train_index, test_index in skf.split(data_file['content'], data_file['label']):
        data_train, dt_test = data_file['content'][train_index], data_file['content'][test_index]
        label_train, label_test = data_file['label'][train_index], data_file['label'][test_index]

        data_train = pd.DataFrame(data_train)

        data_train['label'] = label_train
        data_train.columns = ['content', 'label']
        data_train = data_train.reset_index(drop=True)

        all_feature_freq, data_label = text_word_frequency(data_train, wv_model, fstop_list)
        data_frame = pd.DataFrame(all_feature_freq)
        clf = RandomForestClassifier(n_estimators=79, criterion='gini', max_features='auto')
        clf.fit(data_frame, data_label)

        dt_test = pd.DataFrame(dt_test)
        dt_test['label'] = label_test

        dt_test.columns = ['content', 'label']
        dt_test = dt_test.reset_index(drop=True)

        test_feature, label_test = text_word_frequency(dt_test, wv_model, fstop_list)

        predict_label = clf.predict(pd.DataFrame(test_feature))

        accuracy = metrics.accuracy_score(label_test, predict_label)
        precision = metrics.precision_score(label_test, predict_label)
        recall = metrics.recall_score(label_test, predict_label)
        f1_score = metrics.f1_score(label_test, predict_label)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    print u'平均分类准确率%f:' % np.mean(accuracy_list)
    print u'平均分类精确率%f:' % np.mean(precision_list)
    print u'平均分类召回率%f:' % np.mean(recall_list)
    print u'平均分类f1得分%f:' % np.mean(f1_score_list)

    end_time = time.clock()
    print u'运行程序所花费的时间:', (end_time - start_time)

