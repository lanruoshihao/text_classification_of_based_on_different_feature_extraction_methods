#!usr/bin/python
# -*- coding:utf-8 -*-

import re
import time
import jieba
import codecs
import numpy as np
import pandas as pd
from sklearn import metrics
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def corpus_labels(data_file, fstop_list):
    training_corpus = []
    all_docs = []; docs = []
    train_labels = []
    for index in range(len(data_file)):
        doc_content = data_file.loc[index, 'content']
        if isinstance(doc_content, (float, int)) or len(doc_content) == 0:
            continue
        file_content = re.sub(pattern, '', doc_content)  # 用于训练模型的语料
        train_labels.append(data_file.loc[index, 'label'])
        all_docs.append(str(file_content))

    for i, texts in enumerate(all_docs):
        train_data = jieba.lcut(str(texts))
        train_texts_list = [word for word in train_data if word not in fstop_list and len(word) >= 2]  # 预处理后的训练语料
        document = TaggedDocument(words=train_texts_list, tags=[i])
        training_corpus.append(document)
        docs.append(train_texts_list)
    return training_corpus, train_labels, docs


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


def word_lower_upper_limit(all_word_freq, data_label):
    '''
    计算每个文本中构造的特征的上下限
    :param data:
    :return:
    '''

    df = pd.DataFrame(all_word_freq)
    df['label_cols'] = list(y_train)
    lower_upper_limit = []
    feature_mean = pd.DataFrame()
    for cols_num in range(df.columns.size - 1):
        data_set = df[cols_num].groupby(df['label_cols']).agg(['max', 'min'])  # 根据标签号分组求最大值与最小值
        lower_upper_limit.append(data_set)
        feature_mean[cols_num] = df[cols_num].groupby(df['label_cols']).mean()  # 按行求均值
    return lower_upper_limit, feature_mean


def node_distance(test_pos, pos_center):
    '''
    计算空间中任意两点之间的距离
    :param test_pos:
    :param pos_center:
    :return:
    '''
    dis_vec = np.array(test_pos) - np.array(pos_center)
    euc_dis = np.sqrt(np.sum(np.square(dis_vec)))
    return euc_dis


if __name__ == '__main__':
    start_time = time.clock()
    stopwords_path = 'stoplist.txt'  # 停用词词表
    f_stop = codecs.open(stopwords_path, "rb", 'utf-8')
    fstop_list = f_stop.read()

    pattern = re.compile(u"[\d\s+\.\!\/_,$%^*()?;；。：、:-【】+\"\']+|[+——！，;:。：？、~@#￥%……&* -]+|[()（）～]+")

    file_path = 'origin_dataset.csv'
    data_file = pd.read_csv(file_path)

    training_corpus, train_labels, docs = corpus_labels(data_file, fstop_list)

    size_dim = 600  # 选择特征的维度,可根据需要修改

    model = Doc2Vec(vector_size=size_dim, min_count=1, window=5, epochs=10, workers=3, alpha=0.025)  # 模型的参数
    model.build_vocab(training_corpus)

    x_train, x_test, y_train, y_test = train_test_split(training_corpus, train_labels, test_size=0.2)  # 随机划分训练集和测试集

    model.train(training_corpus, total_examples=len(training_corpus), epochs=model.iter)  # 训练

    train_arrays = getVecs(model, x_train, size_dim)

    data_frame = pd.DataFrame(train_arrays)
    clf = RandomForestClassifier(n_estimators=79, criterion='gini', max_features='auto')
    clf.fit(data_frame, y_train)

    test_arrays = getVecs(model, x_test, size_dim)
    predict_label = clf.predict(pd.DataFrame(test_arrays))

    accuracy = metrics.accuracy_score(y_test, predict_label)
    precision = metrics.precision_score(y_test, predict_label)
    recall = metrics.recall_score(y_test, predict_label)
    f1_score = metrics.f1_score(y_test, predict_label)

    print u'准确率%f:' % accuracy
    print u'精确率%f:' % precision
    print u'召回率%f:' % recall
    print u'f1得分%f:' % f1_score

    end_time = time.clock()
    print u'运行程序所花费的时间:', (end_time - start_time)

