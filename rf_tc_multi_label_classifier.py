# -*- coding:utf-8 -*-

import jieba
import numpy as np
import pandas as pd
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import codecs
import operator
import re
import time


def read_data_train(data, fstop_list):
    # 将类别1到3和5到11视为正例，类别4视为负例
    pos1_words = []; pos2_words = []
    pos3_words = []; pos5_words = []
    pos6_words = []; pos7_words = []
    pos8_words = []; pos9_words = []
    pos10_words = []; pos11_words = []
    neg_words = []

    pattern = re.compile(u"[\d\s+\.\!\/_,$%^*()?;；。：、:-【】+\"\']+|[+——！，;:。：？、~@#￥%……&* -]+|[()（）]")

    for index, value in data.iterrows():
        value_name = data.loc[index, 'content']
        if not (isinstance(value_name, (float, int)) or len(value_name) == 0):
            sentence = jieba.lcut(re.sub(pattern, "", value_name))
        else:
            continue
        content_words = [word for word in sentence if word not in fstop_list and len(word) >= 2]
        # print("content_words:", content_words)
        if int(data.loc[index, u'label']) == 1:
            pos1_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 2:
            pos2_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 3:
            pos3_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 5:
            pos5_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 6:
            pos6_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 7:
            pos7_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 8:
            pos8_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 9:
            pos9_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 10:
            pos10_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 11:
            pos11_words.extend(content_words)
        elif int(data.loc[index, u'label']) == 4:
            neg_words.extend(content_words)  # 负样例
    return pos1_words, pos2_words, pos3_words, pos5_words, pos6_words, pos7_words, \
           pos8_words, pos9_words, pos10_words, pos11_words, neg_words


def create_word_scores(pos1_data, pos2_data, pos3_data, pos5_data, pos6_data, pos7_data,
                       pos8_data, pos9_data, pos10_data, pos11_data, neg_data):
    '''
    计算每个特征的卡方统计量大小
    :param pos1_data:
    :param pos2_data:
    :param pos3_data:
    :param pos5_data:
    :param pos6_data:
    :param pos7_data:
    :param pos8_data:
    :param pos9_data:
    :param pos10_data:
    :param pos11_data:
    :param neg_data:
    :return:
    '''
    # 读取正样本
    pos1Words = list(pos1_data)
    pos2Words = list(pos2_data)
    pos3Words = list(pos3_data)
    pos5Words = list(pos5_data)
    pos6Words = list(pos6_data)
    pos7Words = list(pos7_data)
    pos8Words = list(pos8_data)
    pos9Words = list(pos9_data)
    pos10Words = list(pos10_data)
    pos11Words = list(pos11_data)

    # 读取负样本，其它类别
    negWords = list(neg_data)

    word_fd = FreqDist()        #可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频
    for word in pos1Words:
        word_fd[word] += 1
        cond_word_fd[u'1'][word] += 1

    for word in pos2Words:
        word_fd[word] += 1
        cond_word_fd[u'2'][word] += 1

    for word in pos3Words:
        word_fd[word] += 1
        cond_word_fd[u'3'][word] += 1

    for word in pos5Words:
        word_fd[word] += 1
        cond_word_fd[u'5'][word] += 1

    for word in pos6Words:
        word_fd[word] += 1
        cond_word_fd[u'6'][word] += 1

    for word in pos7Words:
        word_fd[word] += 1
        cond_word_fd[u'7'][word] += 1

    for word in pos8Words:
        word_fd[word] += 1
        cond_word_fd[u'8'][word] += 1

    for word in pos9Words:
        word_fd[word] += 1
        cond_word_fd[u'9'][word] += 1

    for word in pos10Words:
        word_fd[word] += 1
        cond_word_fd[u'10'][word] += 1

    for word in pos11Words:
        word_fd[word] += 1
        cond_word_fd[u'11'][word] += 1

    for word in negWords:
        word_fd[word] += 1
        cond_word_fd[u'4'][word] += 1

    pos1_word_count = cond_word_fd[u'1'].N()  # 积极词1的数量
    pos2_word_count = cond_word_fd[u'2'].N()  # 积极词2的数量
    pos3_word_count = cond_word_fd[u'3'].N()  # 积极词3的数量
    pos5_word_count = cond_word_fd[u'5'].N()  # 积极词5的数量
    pos6_word_count = cond_word_fd[u'6'].N()  # 积极词6的数量
    pos7_word_count = cond_word_fd[u'7'].N()  # 积极词7的数量
    pos8_word_count = cond_word_fd[u'8'].N()  # 积极词8的数量
    pos9_word_count = cond_word_fd[u'9'].N()  # 积极词9的数量
    pos10_word_count = cond_word_fd[u'10'].N()  # 积极词10的数量
    pos11_word_count = cond_word_fd[u'11'].N()  # 积极词11的数量

    neg_word_count = cond_word_fd[u'4'].N()  # 消极词的数量

    total_word_count = pos1_word_count + pos2_word_count + pos3_word_count + pos5_word_count \
                + pos6_word_count + pos7_word_count + pos8_word_count \
                + pos9_word_count + pos10_word_count + pos11_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos1_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'1'][word], (freq, pos1_word_count),
                                                total_word_count)  # 计算积极词的卡方统计量
        pos2_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'2'][word], (freq, pos2_word_count),
                                                total_word_count)
        pos3_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'3'][word], (freq, pos3_word_count),
                                                total_word_count)
        pos5_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'5'][word], (freq, pos5_word_count),
                                                total_word_count)
        pos6_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'6'][word], (freq, pos6_word_count),
                                                total_word_count)
        pos7_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'7'][word], (freq, pos7_word_count),
                                                total_word_count)
        pos8_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'8'][word], (freq, pos8_word_count),
                                                total_word_count)
        pos9_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'9'][word], (freq, pos9_word_count),
                                                total_word_count)
        pos10_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'10'][word], (freq, pos10_word_count),
                                                 total_word_count)
        pos11_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'11'][word], (freq, pos11_word_count),
                                                 total_word_count)

        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd[u'4'][word], (freq, neg_word_count),
                                               total_word_count)
        word_scores[word] = pos1_score + pos2_score + pos3_score + pos5_score + pos6_score + pos7_score + \
                            pos8_score + pos9_score + pos10_score + pos11_score + neg_score   #一个词的信息量等于积极卡方统计量加上消极卡方统计量
    return word_scores


def find_best_words(word_scores, number):
    '''
    将特征按照卡方统计量从大到小排序
    :param word_scores:
    :param number:
    :return:
    '''
    best_vals = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[
                :number]  # 把词按信息量倒序排序。number是特征的维度
    best_words = list([w for w, s in best_vals])
    return best_words


def construct_text_list(text_test, fstop_list):
    '''
    切分文本，并提取出关键词
    :param text_test:
    :return:
    '''
    data_list = jieba.lcut(text_test)
    text_list = [word for word in data_list if word not in fstop_list and len(word) >= 2]
    return text_list


def text_word_frequency(data, data_dict, fstop_list):
    '''
    求出所有训练样本的特征的词频
    :param data:
    :return:
    '''
    all_word_freq = []
    data_label = []
    # data_dict = construct_dict(data)
    for index in range(len(data)):
        data_list = jieba.lcut(data.loc[index, 'content'])
        data_label.append(int(data.loc[index, 'label']))
        text_list = [word for word in data_list if word not in fstop_list and len(word) >= 2]
        word_frequency = []
        for word in data_dict:
            word_count = text_list.count(word)
            word_frequency.append(word_count)
        all_word_freq.append(word_frequency)
    return all_word_freq, data_dict, data_label


if __name__ == '__main__':
    start_time = time.clock()
    stopwords_path = 'stoplist.txt'  # 停用词词表
    f_stop = codecs.open(stopwords_path, "rb", 'utf-8')
    fstop_list = f_stop.read()

    pattern = re.compile(u"[\d\s+\.\!\/_,$%^*()?;；。：、:-【】+\"\']+|[+——！，;:。：？、~@#￥%……&* -]+|[()（）～]+")

    file_path = 'muti_label_origin_dataset.csv'
    data_file = pd.read_csv(file_path, encoding='gb18030')

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []

    skf = StratifiedKFold(n_splits=5)  # 5折交叉验证
    for train_index, test_index in skf.split(data_file['content'], data_file['label']):
        data_train, dt_test = data_file['content'][train_index], data_file['content'][test_index]
        label_train, label_test = data_file['label'][train_index], data_file['label'][test_index]

        data_train = pd.DataFrame(data_train)

        data_train['label'] = label_train
        data_train.columns = ['content', 'label']
        data_set = data_train
        data_set = data_set.reset_index(drop=True)
        pos1_words, pos2_words, pos3_words, pos5_words, pos6_words, pos7_words, pos8_words, \
        pos9_words, pos10_words, pos11_words, neg_words = read_data_train(data_set, fstop_list)

        word_scores = create_word_scores(pos1_words, pos2_words, pos3_words, pos5_words, pos6_words, \
                                         pos7_words, pos8_words, pos9_words, pos10_words, pos11_words, neg_words)

        number = 8000          #选取卡方统计量大小排在前8000个特征
        data_dict = find_best_words(word_scores, number)

        all_word_freq, data_dict, data_label = text_word_frequency(data_set, data_dict, fstop_list)

        data_frame = pd.DataFrame(all_word_freq)
        clf = RandomForestClassifier(n_estimators=59, criterion='gini', max_features='auto')
        clf.fit(data_frame, data_label)

        dt_test = pd.DataFrame(dt_test)

        dt_test.columns = ['content']
        data_test = dt_test
        data_test = data_test.reset_index(drop=True)

        test_result = []
        test_word_fre = []
        for js in range(len(data_test)):
            text_list = construct_text_list(data_test.loc[js, 'content'], fstop_list)
            word_frequency = []
            predict_label = []
            for word in data_dict:
                word_count = text_list.count(word)
                word_frequency.append(word_count)
            test_word_fre.append(word_frequency)

        predict_label = clf.predict(pd.DataFrame(test_word_fre))

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

