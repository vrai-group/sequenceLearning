#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import datetime
import re

filename = "./dataset/data" # Aruba dataset http://ailab.wsu.edu/casas/datasets/

#dateset fields
timestamps = []
sensors = []
values = []
activities = []

activity = ''

print('Loading dataset ...')
with open(filename, 'rb') as features:
    database = features.readlines()
    for line in database: #each line
        f_info = line.split() #find fields
        if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
            f_info[1] = f_info[1] + '.000000'
        timestamps.append(datetime.datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                     "%Y-%m-%d%H:%M:%S.%f"))
        sensors.append(str(np.array(f_info[2])))
        values.append(str(np.array(f_info[3])))

        if len(f_info) == 4:  # if activity exists
            activities.append(activity)
        else:  # if activity does not exist
            des = str(' '.join(np.array(f_info[4:])))
            if 'begin' in des:
                activity = re.sub('begin', '', des)
                if activity[-1] == ' ':
                    activity = activity[:-1]
                activities.append(activity)
            if 'end' in des:
                activities.append(activity)
                activity = ''
features.close()
