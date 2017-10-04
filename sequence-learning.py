#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import datetime
import re
from sklearn.metrics import confusion_matrix, classification_report
from seqlearn.hmm import MultinomialHMM
from seqlearn.evaluation import SequenceKFold

filename = "./dataset/data"  # 15 Milan dataset   http://ailab.wsu.edu/casas/datasets/

def load_dataset():
    # dateset fields
    timestamps = []
    sensors = []
    values = []
    activities = []

    activity = ''  # empty

    print('Loading dataset ...')
    with open(filename, 'rb') as features:
        database = features.readlines()
        for line in database:  # each line
            f_info = line.decode().split()  # find fields
            if 'M' == f_info[2][0] or 'D' == f_info[2][0] or 'T' == f_info[2][0]:
                # choose only M D T sensors, avoiding unexpected errors
                if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                    f_info[1] = f_info[1] + '.000000'
                timestamps.append(datetime.datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                             "%Y-%m-%d%H:%M:%S.%f"))
                sensors.append(str(np.array(f_info[2])))
                values.append(str(np.array(f_info[3])))

                if len(f_info) == 4:  # if activity does not exist
                    activities.append(activity)
                else:  # if activity exists
                    des = str(' '.join(np.array(f_info[4:])))
                    if 'begin' in des:
                        activity = re.sub('begin', '', des)
                        if activity[-1] == ' ':  # if white space at the end
                            activity = activity[:-1]  # delete white space
                        activities.append(activity)
                    if 'end' in des:
                        activities.append(activity)
                        activity = ''
    features.close()

    # dictionaries: assigning keys to values

    temperature = []
    for element in values:
        try:
            temperature.append(float(element))
        except ValueError:
            pass

    sensorsList = sorted(set(sensors))
    dictSensors = {}
    for i, sensor in enumerate(sensorsList):
        dictSensors[sensor] = i
    # print(dictSensors)

    activityList = sorted(set(activities))
    dictActivities = {}
    for i, activity in enumerate(activityList):
        dictActivities[activity] = i
    # print(dictActivities)

    valueList = sorted(set(values))
    dictValues = {}
    for i, v in enumerate(valueList):
        dictValues[v] = i
    # print(dictValues)

    dictObs = {}
    count = 0
    for key in dictSensors.keys():
        if "M" in key:
            dictObs[key + "OFF"] = count
            count += 1
            dictObs[key + "ON"] = count
            count += 1
        if "D" in key:
            dictObs[key + "CLOSE"] = count
            count += 1
            dictObs[key + "OPEN"] = count
            count += 1
        if "T" in key:
            for temp in range(0, int((max(temperature) - min(temperature)) * 2) + 1):
                dictObs[key + str(float(temp / 2.0) + min(temperature))] = count + temp
    # print(dictObs)

    X = []
    Y = []
    for kk, s in enumerate(sensors):
        if "T" in s:
            X.append(dictObs[s + str(round(float(values[kk]), 1))])
        else:
            X.append(dictObs[s + str(values[kk])])
        Y.append(dictActivities[activities[kk]])
    data = [X, Y, dictActivities]
    return data

def seq_lengths(labels):
    lengths = []
    for i, n in enumerate(labels):
       if (i == 0):
          seq = n
          count = 1
       else:
           if (seq == n):
               count += 1
           else:
               if (seq != n):
                   seq = n
                   lengths.append(count)
                   count = 1
       if (i == (len(labels)-1)):
            lengths.append(count)
    return lengths

def dataset_split(train_index, test_index):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for idx in train_index:
        X_train.append(data[0][idx])
        Y_train.append(data[1][idx])
    for idx in test_index:
        X_test.append(data[0][idx])
        Y_test.append(data[1][idx])
    # X_train = np.array(X_train).reshape(-1,1)
    X_tr = np.array(X_train)
    X_train = (((X_tr[:, None] & (1 << np.arange(8)))) > 0).astype(int)  # vector-> binary matrix
    Y_train = np.array(Y_train)
    # X_test = np.array(X_test).reshape(-1,1)
    X_te = np.array(X_test)
    X_test = (((X_te[:, None] & (1 << np.arange(8)))) > 0).astype(int)
    Y_test = np.array(Y_test)
    return [X_train, X_test, Y_train, Y_test]

data = load_dataset()
kf = SequenceKFold(seq_lengths(data[1]),2)
for tuple in kf:
    train_len = tuple[1]
    test_len = tuple[3]
    split = dataset_split(tuple[0], tuple[2])

    #train the model
    clf = MultinomialHMM()
    clf.fit(split[0],split[2],train_len)

    #evaluate the model
    Y_pred = clf.predict(split[1], test_len)
    print('Accuracy:')
    print(clf.score(split[1], split[3], test_len))
    print('Confusion matrix:')
    labels = list(data[2].values())
    print(confusion_matrix(split[3], Y_pred, labels))
    print('Report:')
    target_names = list(data[2].keys())
    print(classification_report(split[3], Y_pred, target_names=target_names))