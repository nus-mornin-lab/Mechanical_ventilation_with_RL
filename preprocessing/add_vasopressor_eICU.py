# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 10:30:21 2020

@author: liuzhuo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def showvalues(values):
    print('missing_rate: %f\nmedian: %.2f, max: %.2f, min: %.2f' % (values.isnull().sum()/values.shape[0], values.median(), values.max(), values.min()))
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(values[~pd.isnull(values)], vert=False, showfliers=False)
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(values[~pd.isnull(values)], vert=False, showfliers=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(values[~pd.isnull(values)])

data_dir = '../eICU_data_v1'
output_dir = '../eICU_v1'

actions = pd.read_csv(os.path.join(output_dir, 'data_rl.csv'), engine='python')

# use values between time-120 and time+120. 
def get_vital(vital_name, time_name='chartoffset', mode='nearest'):
    temp = vital[~pd.isnull(vital[vital_name])][['patientunitstayid',time_name,vital_name]].reset_index(drop=True)
    temp = temp.sort_values(by=['patientunitstayid',time_name])
    def get_vital_result(data):
        # 4h (240 minutes) per step
        patientno = data['patientunitstayid'].values[0]
        ptemp = temp[temp['patientunitstayid']==patientno]
        if ptemp.shape[0] == 0:
            return pd.DataFrame([[patientno,0,0]], columns=['patientunitstayid','time',vital_name])
        result = pd.merge(data, ptemp, on='patientunitstayid', how='outer')
        result['time_offset'] = abs(result['time']-result[time_name])
        if mode=='nearest':
            result = result.sort_values(by=['time','time_offset'])
            result = result.groupby(['patientunitstayid','time']).first().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='mean':
            result = result[result['time_offset']<=120]
            if result.shape[0]==0:
                return pd.DataFrame([[patientno,0,0]], columns=['patientunitstayid','time',vital_name])
            result = result.groupby(['patientunitstayid','time']).mean().reset_index()
        elif mode=='max':
            result = result[result['time_offset']<=120]
            result = result.groupby(['patientunitstayid','time']).max().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='sum':
            result = result[result['time_offset']<=120]
            result = result.groupby(['patientunitstayid','time']).sum().reset_index().drop([time_name,'time_offset'],axis=1)
        return result
    new_vital = actions[['patientunitstayid','time']].groupby('patientunitstayid').apply(get_vital_result).reset_index(drop=True)
    print(new_vital)
    new_actions = pd.merge(actions, new_vital, on=['patientunitstayid','time'], how='left')
    return new_actions

## vasopressor_edit
vital = pd.read_csv(os.path.join(data_dir, 'pivot_vasopressor_edit.csv'), engine='python')
vital = vital[vital['equivalent_mg_4h'].apply(lambda x: x!='na')]
vital['equivalent_mg_4h'] = vital['equivalent_mg_4h'].apply(lambda x: float(x))
# dose each 4h
old_vital = vital[['patientunitstayid','drugstartoffset','drugstopoffset','equivalent_mg_4h']].reset_index(drop=True)
old_vital = old_vital.sort_values(by=['patientunitstayid','drugstartoffset'])
old_vital = old_vital.reset_index()
def get_vasopressor_result(data):
    # 2h (240 minutes) per step, this will not miss
    result = pd.DataFrame(list(range(int(data['drugstartoffset'].values[0]), int(data['drugstopoffset'].values[0]), 120)),columns=['drug_time'])
    result['patientunitstayid'] = data['patientunitstayid'].values[0]
    result['equivalent_mg_4h'] = data['equivalent_mg_4h'].values[0]
    return result
vital = old_vital.groupby('index').apply(get_vasopressor_result)
vital = vital.sort_values(by=['patientunitstayid','drug_time']).reset_index(drop=True)

actions = get_vital('equivalent_mg_4h', time_name='drug_time', mode='mean').reset_index(drop=True)

# fill na
actions = actions.fillna({'equivalent_mg_4h':0})

# normalize log
origin_rename = {'equivalent_mg_4h':'ori_equivalent_mg_4h'}
actions = actions.rename(columns=origin_rename)
# use log(x+1) and do normalization
actions['equivalent_mg_4h'] = actions['ori_equivalent_mg_4h'].apply(lambda x: (np.log(x+1)-0) / (4.8-0))

# next state
next_rename = {'equivalent_mg_4h':'next_equivalent_mg_4h'}
next_data = actions[['patientunitstayid','step_id','equivalent_mg_4h']].copy().rename(columns=next_rename)
next_data['step_id'] = next_data['step_id'].apply(lambda x: x-1)
actions = pd.merge(actions, next_data, on=['patientunitstayid','step_id'], how='left')

# save data
actions.to_csv(os.path.join(output_dir,'data_rl_with_dose.csv'), index=False)
