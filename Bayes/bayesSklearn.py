#coding=utf-8
import numpy as np
import os
import jieba
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import random

def readData(path, testRatio = 0.2):
    pathList = os.listdir(path)
    dataSet = []; category = []
    for folder in pathList:
        if folder[0] == '.':
            continue
        new_folder_path = os.path.join(path, folder)
        files = os.listdir(new_folder_path)
        j = 1
        for fileName in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path, fileName), 'r') as f:
                raw = f.read()
            words = jieba.cut(raw, cut_all = False)
            dataSet.append(list(words))
            category.append(folder)
            j += 1

    data_class_list = list(zip(dataSet, category))
    random.shuffle(data_class_list)
    testIndex = int(len(data_class_list) * testRatio) + 1
    train_class_list = data_class_list[testIndex:]
    test_class_list = data_class_list[:testIndex]
    train_words_list, train_labels_list = zip(*train_class_list)
    test_words_list, test_labels_list = zip(*test_class_list)

    all_word_dict = {}
    for train_words in train_words_list:
        for word in train_words:
            if word in all_word_dict.keys():
                all_word_dict[word] += 1
            else:
                all_word_dict[word] = 1
    
    tmpList = sorted(all_word_dict.items(), key = lambda x : x[1], reverse = True)
    all_word_list, all_word_count = zip(*tmpList)
    all_word_list = list(all_word_list)
    return all_word_list, train_words_list, train_labels_list, test_words_list, test_labels_list

def getWordSet(filePath):
    wordSet = set()
    with open(filePath, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                wordSet.add(word)
    return wordSet

def getFeatWords(all_word_list, deleteN, wordSet = set()):
    featWords = []
    n = 1
    for i in range(deleteN, len(all_word_list), 1):
        if n > 1000:
            break
        if not all_word_list[i].isdigit() and all_word_list[i] not in wordSet and 1 < len(all_word_list[i]) < 5:
            featWords.append(all_word_list[i])
        n += 1
    return featWords

def words2Vec(train_words_list, test_words_list, featWords):
    def text_features(text, featWords):
        wordSet = set(text)
        features = [1 if word in wordSet else 0 for word in featWords]
        return features
    train_feature_list = [text_features(text, featWords) for text in train_words_list]
    test_feature_list = [text_features(text, featWords) for text in test_words_list]
    return train_feature_list, test_feature_list

def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    path = './SogouC/Sample'
    all_word_list, train_words_list, train_labels_list, test_words_list, test_labels_list = readData(path)
    
    stopwordsPath = './stopwords_cn.txt'
    stopwordSet = getWordSet(stopwordsPath)

    test_accuracy_list = []
    # deleteNs = range(0, 1000, 20)                #0 20 40 60 ... 980
    # for deleteN in deleteNs:
        # feature_words = getFeatWords(all_word_list, deleteN, stopwordSet)
        # train_feature_list, test_feature_list = words2Vec(train_words_list, test_words_list, feature_words)
        # test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_labels_list, test_labels_list)
        # test_accuracy_list.append(test_accuracy)
 
    feature_words = getFeatWords(all_word_list, 500, stopwordSet)
    train_feature_list, test_feature_list = words2Vec(train_words_list, test_words_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_labels_list, test_labels_list)
    test_accuracy_list.append(test_accuracy)

    # plt.figure()
    # plt.plot(deleteNs, test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.show()

    ave = lambda c: sum(c) / len(c)

    print(ave(test_accuracy_list))