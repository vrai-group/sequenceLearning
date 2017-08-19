#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import datetime
import re
from sklearn.metrics import confusion_matrix
from seqlearn.hmm import MultinomialHMM
from seqlearn.evaluation import SequenceKFold

filename = "./dataset/data"  # 15 Milan dataset   http://ailab.wsu.edu/casas/datasets/

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
        if ('M' == f_info[2][0] or 'D' == f_info[2][0] or 'T' == f_info[2][
            0]):  # choose only M D T sensors, avoiding unexpected errors
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
#print(dictSensors)

activityList = sorted(set(activities))
dictActivities = {}
for i, activity in enumerate(activityList):
    dictActivities[activity] = i

valueList = sorted(set(values))
dictValues = {}
for i, v in enumerate(valueList):
    dictValues[v] = i
#print(dictValues)

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

#print (dictObs)
X = []
Y = []
for kk, s in enumerate(sensors):
    if "T" in s:
        X.append(dictObs[s + str(round(float(values[kk]), 1))])
    else:
        X.append(dictObs[s + str(values[kk])])
    Y.append(dictActivities[activities[kk]])

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
       if (i == (len(Y)-1)):
            lengths.append(count)
    return lengths

print('Creating folds ...')
kf = SequenceKFold(seq_lengths(Y),20)
for tuple in kf:
    train_index = tuple[0]
    train_len = tuple[1]
    test_index = tuple[2]
    test_len = tuple[3]
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for idx in train_index:
        X_train.append(X[idx])
        Y_train.append(Y[idx])
    for idx in test_index:
        X_test.append(X[idx])
        Y_test.append(Y[idx])
    X_train = np.array(X_train).reshape(-1,1)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test).reshape(-1,1)
    Y_test = np.array(Y_test)
    clf = MultinomialHMM()
    clf.fit(X_train,Y_train,train_len)
    Y_pred = clf.predict(X_test,test_len)
    print('Train set:')
    print(Y_train)
    print('Test set:')
    print(Y_test)
    print('Prediction:')
    print(Y_pred)
    print('Confusion matrix:')
    print(confusion_matrix(Y_test,Y_pred))