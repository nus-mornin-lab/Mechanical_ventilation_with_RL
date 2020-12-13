# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 10:30:21 2020

@author: liuzhuo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def showvalues(values):
    print('missing_rate: %f\nmedian: %.2f, max: %.2f, min: %.2f' % (values.isnull().sum()/values.shape[0], values.median(), values.max(), values.min()))
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(values[~pd.isnull(values)], vert=False, showfliers=False)
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(values[~pd.isnull(values)], vert=False, showfliers=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(values[~pd.isnull(values)])

data_dir = '../eICU_data_v1'
output_dir = '../eICU_v2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## patients
patients = pd.read_csv(os.path.join(data_dir,'cohort.csv'), engine='python')
demo = pd.read_csv(os.path.join(data_dir,'demographic.csv'), engine='python')
patients = pd.merge(patients, demo, on='patientunitstayid', how='left')

#patients['vent_time'] = patients['vent_end']-patients['vent_start']

## join actions, remain 37547
actions = pd.read_csv(os.path.join(data_dir,'pivot_action.csv'), engine='python')
patients = patients[patients['patientunitstayid'].isin(actions.patientunitstayid)]

# age, exclude >89 and < 16, remain 36441
patients = patients[patients['age'].apply(lambda x: (False if pd.isnull(x) else False if int(x)<16 else True) if '>' not in str(x) else False)]
patients['age'] = patients['age'].apply(lambda x: int(x))

# height, exclude >=250 or <=130, remain 36094
patients = patients[patients['admissionheight'].apply(lambda x: True if pd.isnull(x) else x>130 and x<250)]
patients = patients[patients['admissionheight'].apply(lambda x: not pd.isnull(x))]

# gender, # 0 female, 1 male, exclude null, remain 36091
#print(patients.gender.value_counts())
#values = patients['gender']
#print('missing_rate: %f\nmedian: %.2f, max: %.2f, min: %.2f' % (values.isnull().sum()/values.shape[0], values.median(), values.max(), values.min()))
patients = patients[patients['gender'].apply(lambda x: not pd.isnull(x))]

# Ideal body weight is computed in men as 50 + (0.91 × [height in centimeters − 152.4]) and in women as 45.5 + (0.91 × [height in centimeters − 152.4])
def idealweight(x):
    if x['gender']==1: #male
        result = 50 + 0.91 * (x['admissionheight']-152.4)
    else:
        result = 45.5 + 0.91 * (x['admissionheight']-152.4)
    return result
patients['idealweight'] = patients.apply(idealweight, axis=1)
#showvalues(patients['idealweight'])

print('patients remaining', patients.shape[0])

# new vent start and vent end
action_patients = actions.groupby('patientunitstayid')['respchartoffset'].max().reset_index().rename(columns={'respchartoffset':'vent_end'})
action_patients = pd.merge(action_patients, actions.groupby('patientunitstayid')['respchartoffset'].min().reset_index().rename(columns={'respchartoffset':'vent_start'}),
                           on='patientunitstayid', how='left')
action_patients['vent_time'] = action_patients['vent_end'] - action_patients['vent_start']

# exclude too long or too short vent_time (<=20000,>=44*60)
remain_patients = action_patients[action_patients['vent_time'].apply(lambda x: x<=20000)]
remain_patients = action_patients[action_patients['vent_time'].apply(lambda x: x>=2640)]
remain_patients['vent_end'] = remain_patients.apply(lambda x: x['vent_start']+2880 if x['vent_time']<2880 else x['vent_end'], axis=1)

actions = pd.merge(actions, remain_patients, on='patientunitstayid', how='left')
actions = actions[~pd.isnull(actions['vent_time'])]

actions['PEEP'] = actions.apply(lambda x: x['PEEP_2'] if pd.isnull(x['PEEP_1']) else x['PEEP_1'], axis=1)
actions['PEEP'] = actions['PEEP'].apply(lambda x: float(x) if x!='100%' else np.nan)

# FiO2_1 is in the right range, but too many missing
actions['FiO2'] = actions.apply(lambda x: (x['FiO2_3'] if pd.isnull(x['FiO2_2']) else x['FiO2_2']) if pd.isnull(x['FiO2_1']) else x['FiO2_1'] , axis=1)
actions['FiO2'] = actions['FiO2'].apply(lambda x: np.nan if pd.isnull(x) else float(x) if '%' not in str(x) else float(x.replace('%','')))

actions['Tidal_volume'] = actions.apply(lambda x: (x['Tidal_volume_3'] if pd.isnull(x['Tidal_volume_2']) else x['Tidal_volume_2']) if pd.isnull(x['Tidal_volume_1']) else x['Tidal_volume_1'] , axis=1)
actions['Tidal_volume'] = actions['Tidal_volume'].apply(lambda x: float(x) if x!='400%' else np.nan)

print('action rows', actions.shape[0])

actions = pd.merge(actions, patients[['patientunitstayid','idealweight']], on='patientunitstayid', how='left')
actions = actions[~pd.isnull(actions['idealweight'])]

# Tidal to ml/kg ideal weight
actions['Tidal'] = actions.apply(lambda x: np.nan if pd.isnull(x['Tidal_volume']) else x['Tidal_volume']/x['idealweight'], axis=1)

# remove useless columns
actions = actions.drop(['PEEP_1', 'PEEP_2', 'FiO2_1', 'FiO2_2', 'FiO2_3', 'Tidal_volume_1',
        'Tidal_volume_2', 'Tidal_volume_3', 'Tidal_volume'], axis=1)

# set abnormal values to null
abnormals = {} # min, max
abnormals['PEEP'] = [0, 25]
abnormals['FiO2'] = [20, 100]
abnormals['Tidal'] = [3.5, 25]
def remove_abnormal(data, col):
    data[col] = data[col].apply(lambda x: np.nan if x<abnormals[col][0] or x>abnormals[col][1] else x)
remove_abnormal(actions, 'PEEP')
remove_abnormal(actions, 'FiO2')
remove_abnormal(actions, 'Tidal')
# show actions
#showvalues(actions['PEEP'])
#showvalues(actions['FiO2'])
#showvalues(actions['Tidal'])

print('action rows after join patients and remove wrong time', actions.shape[0])
# remain 24642
print('action patients number after above', actions.patientunitstayid.unique().shape)

# split actions into several steps
#TODO 2h per step
time_span = 240
def get_actions(action_type):
    temp = actions[~pd.isnull(actions[action_type])][['patientunitstayid','respchartoffset',action_type,'vent_start','vent_end']].reset_index(drop=True)
    temp = temp.sort_values(by=['patientunitstayid','respchartoffset'])
    def get_action_result(data):
        
        result = pd.DataFrame(list(range(int(data.vent_start.values[0])+time_span//2, int(data.vent_end.values[0]), time_span)),columns=['time'])
        result['patientunitstayid'] = data['patientunitstayid'].values[0]
        result = pd.merge(result, data[['patientunitstayid','respchartoffset',action_type]], on='patientunitstayid', how='outer')
        result['time_offset'] = abs(result['time']-result['respchartoffset'])
        result = result[result['time_offset']<=time_span/2]
        result = result.groupby(['patientunitstayid','time']).mean().reset_index().drop(['respchartoffset','time_offset'],axis=1)
        return result
    new_actions = temp.groupby('patientunitstayid').apply(get_action_result)
    return new_actions

peep_actions = get_actions('PEEP').drop(['patientunitstayid'],axis=1).reset_index().rename(columns={'level_1':'step_id'})
fio2_actions = get_actions('FiO2').drop(['patientunitstayid'],axis=1).reset_index().rename(columns={'level_1':'step_id'})
tidal_actions = get_actions('Tidal').drop(['patientunitstayid'],axis=1).reset_index().rename(columns={'level_1':'step_id'})

old_actions = actions.copy()
del actions
def time_actions(data):
    result = pd.DataFrame(list(range(int(data.vent_start.values[0])+time_span//2, int(data.vent_end.values[0]), time_span)),columns=['time'])
    result['patientunitstayid'] = data['patientunitstayid'].values[0]
    return result
actions = old_actions.groupby('patientunitstayid').apply(time_actions)
actions = actions.drop(['patientunitstayid'],axis=1).reset_index().rename(columns={'level_1':'step_id'})

actions = pd.merge(actions, peep_actions, on=['patientunitstayid','step_id','time'], how='left')
actions = pd.merge(actions, fio2_actions, on=['patientunitstayid','step_id','time'], how='left')
actions = pd.merge(actions, tidal_actions, on=['patientunitstayid','step_id','time'], how='left')

# action missing rate
value_missing = actions.groupby('patientunitstayid').apply(lambda x: x[x.apply(lambda a: pd.isnull(a['PEEP']) and pd.isnull(a['FiO2']) and pd.isnull(a['Tidal']), axis=1)].shape[0]/x.shape[0])
value_missing = value_missing.reset_index().rename(columns={0:'action_missing_rate'}) # median 
value_missing.to_csv(os.path.join(output_dir,'action_missing_rate.csv'), index=False)

# missing < 15%, we keep the missing values
#value_missing = value_missing[value_missing['action_missing_rate']<0.15]
#actions = actions[actions['patientunitstayid'].isin(value_missing.patientunitstayid)]

# use last to fill na
actions = actions.sort_values(by=['patientunitstayid','step_id'])
actions = actions.reset_index(drop=True)

def fill_using_up_and_down(x, action_type):
    na_values = x[pd.isnull(x[action_type])][['patientunitstayid','step_id']]
    real_values = x[~pd.isnull(x[action_type])][['patientunitstayid','step_id',action_type]].rename(columns={'step_id':'real_id'})
    if real_values.shape[0]==0 or na_values.shape[0]==0:
        return pd.DataFrame()
    # using the last step
    values = pd.merge(na_values, real_values, on='patientunitstayid', how='left')
    values['id_interval'] = values['step_id'] - values['real_id']
    values = values[values['id_interval']>0]
    values = values.sort_values(by=['patientunitstayid','step_id','id_interval'])
    values = values.groupby(['patientunitstayid','step_id']).first().reset_index()
    # using the next step
    not_na_values = pd.merge(na_values, values[['patientunitstayid','step_id',action_type]], on=['patientunitstayid','step_id'], how='left')
    still_na_values = not_na_values[pd.isnull(not_na_values[action_type])][['patientunitstayid','step_id']]
    if still_na_values.shape[0] == 0:
        return not_na_values
    values = pd.merge(still_na_values, real_values, on='patientunitstayid', how='left')
    values['id_interval'] = values['real_id'] - values['step_id'] # opposite
    values = values[values['id_interval']>0]
    values = values.sort_values(by=['patientunitstayid','step_id','id_interval'])
    values = values.groupby(['patientunitstayid','step_id']).first().reset_index()
    still_na_values = pd.merge(still_na_values, values[['patientunitstayid','step_id',action_type]], on=['patientunitstayid','step_id'], how='left')
    not_na_values = pd.concat([not_na_values, still_na_values])
    return not_na_values

for action_type in ['PEEP','Tidal','FiO2']:
    fills = actions.groupby('patientunitstayid').apply(lambda x: fill_using_up_and_down(x, action_type)).reset_index(drop=True)
    if fills.shape[0] != 0:
        actions = pd.merge(actions.rename(columns={action_type:'real_'+action_type}), fills.rename(columns={action_type:'fill_'+action_type}), on=['patientunitstayid','step_id'], how='left')
        actions[action_type] = actions.apply(lambda x: x['real_'+action_type] if not pd.isnull(x['real_'+action_type]) else x['fill_'+action_type], axis=1)
        actions = actions.drop(['fill_'+action_type,'real_'+action_type], axis=1)

'''
# Too slow
for i in range(1, actions.shape[0]):
    if actions.loc[i,'patientunitstayid']==actions.loc[i-1,'patientunitstayid']:
        if pd.isnull(actions.loc[i,'PEEP']):
            actions.loc[i,'PEEP'] = actions.loc[i-1,'PEEP']
        if pd.isnull(actions.loc[i,'FiO2']):
            actions.loc[i,'FiO2'] = actions.loc[i-1,'FiO2']
        if pd.isnull(actions.loc[i,'Tidal']):
            actions.loc[i,'Tidal'] = actions.loc[i-1,'Tidal']

for i in range(actions.shape[0]-2,-1,-1):
    if actions.loc[i,'patientunitstayid']==actions.loc[i+1,'patientunitstayid']:
        if pd.isnull(actions.loc[i,'PEEP']):
            actions.loc[i,'PEEP'] = actions.loc[i+1,'PEEP']
        if pd.isnull(actions.loc[i,'FiO2']):
            actions.loc[i,'FiO2'] = actions.loc[i+1,'FiO2']
        if pd.isnull(actions.loc[i,'Tidal']):
            actions.loc[i,'Tidal'] = actions.loc[i+1,'Tidal']
'''
# do not fill others using PEEP=5, FiO2=40, Tidal=7.5
#actions = actions.fillna({'PEEP':5,'FiO2':40, 'Tidal':7.5})

# remove 100% missing of each action
actions = actions[~actions.apply(lambda x: pd.isnull(x['PEEP']) or pd.isnull(x['FiO2']) or pd.isnull(x['Tidal']), axis=1)]

# save temperory result of actions
actions.to_csv(os.path.join(output_dir,'actions_4h.csv'), index=False)
patients.to_csv(os.path.join(output_dir,'remain_patients.csv'), index=False)

print('action rows remove too much missing', actions.shape[0])
print('action patients number after above', actions.patientunitstayid.unique().shape)

# action for one patient
def showonepatient(pid):                                                                                                            
    peep = {}
    fio2 = {}
    tidal_ten = {}
    p1 = actions[actions['patientunitstayid']==pid]
    for i in p1.index:
        if not pd.isnull(p1.loc[i,'PEEP']):
            peep[p1.loc[i,'time']/240] = p1.loc[i,'PEEP']
        if not pd.isnull(p1.loc[i,'FiO2']):
            fio2[p1.loc[i,'time']/240] = p1.loc[i,'FiO2']/10
        if not pd.isnull(p1.loc[i,'Tidal']):
            tidal_ten[p1.loc[i,'time']/240] = p1.loc[i,'Tidal']
    fig, ax = plt.subplots(figsize=(30,5))
    ax.scatter(peep.keys(), peep.values(), color='green', label='PEEP')
    ax.scatter(fio2.keys(), fio2.values(), color='red', label='FiO2')
    ax.scatter(tidal_ten.keys(), tidal_ten.values(), color='blue', label='Tidal')
    ax.set_xlabel('time (4h)')
    ax.legend()
    ax.grid(axis='x')
#showonepatient(141515)
