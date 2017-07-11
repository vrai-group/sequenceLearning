#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import datetime
import re

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
print(dictSensors)

activityList = sorted(set(activities))
dictActivities = {}
for i, activity in enumerate(activityList):
    dictActivities[activity] = i

valueList = sorted(set(values))
dictValues = {}
for i, v in enumerate(valueList):
    dictValues[v] = i
print(dictValues)

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

print (dictObs)
X = []
Y = []
for kk, s in enumerate(sensors):
    if "T" in s:
        X.append(dictObs[s + str(round(float(values[kk]), 1))])
    else:
        X.append(dictObs[s + str(values[kk])])
    Y.append(dictActivities[activities[kk]])
