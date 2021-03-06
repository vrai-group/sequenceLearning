#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import datetime
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

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

def dataset_split(X, Y, train_perc):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    max_lenght = 0
    for i, n in enumerate(Y):
        if i == 0:
            seq = n
            count = 1
            a = i
        else:
            if seq == n:
                count += 1
            else:
                if seq != n:
                    if np.random.rand() < train_perc:
                        X_train.append(X[a:a+count])
                        Y_train.append(seq)
                    else:
                        X_test.append(X[a:a+count])
                        Y_test.append(seq)
                    if count > max_lenght:
                        max_lenght = count
                    seq = n
                    count = 1
                    a = i
        if i == (len(Y) - 1):
            if np.random.rand() < train_perc:
                X_train.append(X[a:a + count])
                Y_train.append(seq)
            else:
                X_test.append(X[a:a + count])
                Y_test.append(seq)
            if count > max_lenght:
                max_lenght = count
    # same lenght for all sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_lenght)
    X_test = sequence.pad_sequences(X_test, maxlen=max_lenght)
    split = [X_train, X_test, Y_train, Y_test, max_lenght]
    return split

def create_model(X_train, X_test, max_lenght, dictActivities):
    model = Sequential()
    model.add(Embedding(len(X_train)+len(X_test), 32, input_length=max_lenght))
    model.add(LSTM(100))
    model.add(Dense(len(dictActivities), activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

data = load_dataset()
split = dataset_split(data[0], data[1], 0.8) #train_perc from 0 to 1
model = create_model(split[0], split[1], split[4], data[2])

# train the model
print("Begin training ...")
class_weight = compute_class_weight('balanced', np.unique(data[1]), data[1]) #use as optional argument in the fit function
model.fit(split[0], split[2], validation_split=0.2, epochs=10, batch_size=64)

# evaluate the model
print("Begin testing ...")
scores = model.evaluate(split[1], split[3], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
Y_pred = model.predict_classes(split[1], verbose=0)
print(confusion_matrix(split[3], Y_pred))
target_names = list(data[2].keys())
print(classification_report(split[3], Y_pred, target_names=target_names))