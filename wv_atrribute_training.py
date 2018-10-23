#!usr/bin/python
# -*- coding:utf-8 -*-

from gensim.models import word2vec
import jieba
import pandas as pd
import codecs
import re


stopwords_path = 'stoplist.txt'  # 停用词词表
f_stop = codecs.open(stopwords_path, "rb", 'utf-8')
fstop_list = f_stop.read()

pattern = re.compile(u"[\d\s+\.\!\/_,$%^*()?;；。：、:-【】+\"\']+|[+——！，;:。：？、~@#￥%……&* -]+|[()（）～]+")

file_path = 'origin_dataset.csv'
data_file = pd.read_csv(file_path, encoding='gb18030')

training_corpus = []
for js in range(len(data_file)):
    pre_data = re.sub(pattern, '', data_file.loc[js, 'content']) #用于训练模型的语料
    data_list = jieba.lcut(pre_data)
    train_data = [word for word in data_list if word not in fstop_list and len(word) >= 2]     #预处理后的训练语料
    training_corpus.append(train_data)

wv_model = word2vec.Word2Vec(training_corpus, size=1000, window=5, min_count=1, workers=4)

wv_model.save('word2vec_model.txt')

