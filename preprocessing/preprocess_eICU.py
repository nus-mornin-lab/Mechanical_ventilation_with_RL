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
output_dir = '../eICU_v1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## patients
patients = pd.read_csv(os.path.join(data_dir,'cohort.csv'), engine='python')
demo = pd.read_csv(os.path.join(data_dir,'demographic.csv'), engine='python')
patients = pd.merge(patients, demo, on='patientunitstayid', how='left')

#patients['vent_time'] = patients['vent_end']-patients['vent_start']

# age, exclude >89 and < 16, from 54728 to 52715
patients = patients[patients['age'].apply(lambda x: (False if pd.isnull(x) else False if int(x)<16 else True) if '>' not in str(x) else False)]
patients['age'] = patients['age'].apply(lambda x: int(x))
#showvalues(patients['age'])

# weight
#showvalues(patients['admissionweight'])

# height, exclude >=250 or <=130, from 52715 to ?
#showvalues(patients['admissionheight'])
patients = patients[patients['admissionheight'].apply(lambda x: True if pd.isnull(x) else x>130 and x<250)]
patients = patients[patients['admissionheight'].apply(lambda x: not pd.isnull(x))]

# gender, # 0 female, 1 male, exclude null, from ? to 52092
#print(patients.gender.value_counts())
#values = patients['gender']
#print('missing_rate: %f\nmedian: %.2f, max: %.2f, min: %.2f' % (values.isnull().sum()/values.shape[0], values.median(), values.max(), values.min()))
patients = patients[patients['gender'].apply(lambda x: not pd.isnull(x))]

# exclude apache==-1
patients = patients[patients['apache_iv'].apply(lambda x: x!=-1)]

# Ideal body weight is computed in men as 50 + (0.91 × [height in centimeters − 152.4]) and in women as 45.5 + (0.91 × [height in centimeters − 152.4])
def idealweight(x):
    if x['gender']==1: #male
        result = 50 + 0.91 * (x['admissionheight']-152.4)
    else:
        result = 45.5 + 0.91 * (x['admissionheight']-152.4)
    return result
patients['idealweight'] = patients.apply(idealweight, axis=1)
#showvalues(patients['idealweight'])

# apache iv, apache_iv==apache_iv_1
#showvalues(patients['apache_iv'])

# Modified SOFA, sofatotal
#showvalues(patients['sofatotal'])

print('patients remaining', patients.shape[0])

## actions
actions = pd.read_csv(os.path.join(data_dir,'pivot_action.csv'), engine='python')

action_patients = actions.groupby('patientunitstayid')['respchartoffset'].max().reset_index().rename(columns={'respchartoffset':'vent_end'})
action_patients = pd.merge(action_patients, actions.groupby('patientunitstayid')['respchartoffset'].min().reset_index().rename(columns={'respchartoffset':'vent_start'}),
                           on='patientunitstayid', how='left')
action_patients['vent_time'] = action_patients['vent_end'] - action_patients['vent_start']

# exclude too long or too short vent_time (<=25000,>=44*60)
remain_patients = action_patients[action_patients['vent_time'].apply(lambda x: x<=25000 and x>=2640)]
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
print('action patients number after above', actions.patientunitstayid.unique().shape)

# split actions into several 4h
def get_actions(action_type):
    temp = actions[~pd.isnull(actions[action_type])][['patientunitstayid','respchartoffset',action_type,'vent_start','vent_end']].reset_index(drop=True)
    temp = temp.sort_values(by=['patientunitstayid','respchartoffset'])
    def get_action_result(data):
        # 4h (240 minutes) per step
        result = pd.DataFrame(list(range(int(data.vent_start.values[0])+120, int(data.vent_end.values[0]), 240)),columns=['time'])
        result['patientunitstayid'] = data['patientunitstayid'].values[0]
        result = pd.merge(result, data[['patientunitstayid','respchartoffset',action_type]], on='patientunitstayid', how='outer')
        result['time_offset'] = abs(result['time']-result['respchartoffset'])
        result = result[result['time_offset']<=120]
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
    result = pd.DataFrame(list(range(int(data.vent_start.values[0])+120, int(data.vent_end.values[0]), 240)),columns=['time'])
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

# missing < 50%
value_missing = value_missing[value_missing['action_missing_rate']<0.50]
actions = actions[actions['patientunitstayid'].isin(value_missing.patientunitstayid)]

# use last to fill na
actions = actions.sort_values(by=['patientunitstayid','step_id'])
actions = actions.reset_index(drop=True)
for i in range(1, actions.shape[0]):
    if actions.loc[i,'patientunitstayid']==actions.loc[i-1,'patientunitstayid']:
        if pd.isnull(actions.loc[i,'PEEP']):
            actions.loc[i,'PEEP'] = actions.loc[i-1,'PEEP']
        if pd.isnull(actions.loc[i,'FiO2']):
            actions.loc[i,'FiO2'] = actions.loc[i-1,'FiO2']
        if pd.isnull(actions.loc[i,'Tidal']):
            actions.loc[i,'Tidal'] = actions.loc[i-1,'Tidal']

# fill others using PEEP=5, FiO2=40, Tidal=7.5
actions = actions.fillna({'PEEP':5,'FiO2':40, 'Tidal':7.5})

# save temperory result of actions
actions.to_csv(os.path.join(output_dir,'actions_4h.csv'), index=False)

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

## vital signs
vital = pd.read_csv(os.path.join(data_dir, 'pivot_vital.csv'), engine='python')

vital['sbp'] = vital.apply(lambda x: x['nibp_systolic'] if pd.isnull(x['ibp_systolic']) else x['ibp_systolic'], axis=1)
vital['dbp'] = vital.apply(lambda x: x['nibp_diastolic'] if pd.isnull(x['ibp_diastolic']) else x['ibp_diastolic'], axis=1)
vital['mbp'] = vital.apply(lambda x: x['nibp_mean'] if pd.isnull(x['ibp_mean']) else x['ibp_mean'], axis=1)
vital = vital.drop(['nibp_systolic', 'nibp_diastolic', 'nibp_mean', 'ibp_systolic', 'ibp_diastolic', 'ibp_mean'], axis=1)

# use values between time-120 and time+120. 
def get_vital(vital_name, time_name='chartoffset', mode='nearest'):
    temp = vital[~pd.isnull(vital[vital_name])][['patientunitstayid',time_name,vital_name]].reset_index(drop=True)
    temp = temp.sort_values(by=['patientunitstayid',time_name])
    def get_vital_result(data):
        # 4h (240 minutes) per step
        patientno = data['patientunitstayid'].values[0]
        ptemp = temp[temp['patientunitstayid']==patientno]
        result = pd.merge(data, ptemp, on='patientunitstayid', how='outer')
        result['time_offset'] = abs(result['time']-result[time_name])
        if mode=='nearest':
            result = result.sort_values(by=['time','time_offset'])
            result = result.groupby(['patientunitstayid','time']).first().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='mean':
            result = result[result['time_offset']<=120]
            result = result.groupby(['patientunitstayid','time']).mean().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='max':
            result = result[result['time_offset']<=120]
            result = result.groupby(['patientunitstayid','time']).max().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='sum':
            result = result[result['time_offset']<=120]
            result = result.groupby(['patientunitstayid','time']).sum().reset_index().drop([time_name,'time_offset'],axis=1)
        return result
    new_vital = actions[['patientunitstayid','time']].groupby('patientunitstayid').apply(get_vital_result).reset_index(drop=True)
    new_actions = pd.merge(actions, new_vital, on=['patientunitstayid','time'], how='left')
    return new_actions

for vital_name in ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp']:
    actions = get_vital(vital_name).reset_index(drop=True)

## lab test
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_lab.csv'), engine='python')

for vital_name in ['lactate', 'bicarbonate', 'wbc', 'pao2', 'paco2', 'pH']:
    actions = get_vital(vital_name).reset_index(drop=True)

## gcs
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_gcs.csv'), engine='python')

for vital_name in ['gcs']:
    actions = get_vital(vital_name).reset_index(drop=True)
    
## fluid
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_fluid.csv'), engine='python')
#TODO intaketotal seems not cumulative value
vital = vital.sort_values(by=['patientunitstayid', 'intakeoutputoffset'])

for vital_name in ['intaketotal','nettotal']:
    actions = get_vital(vital_name, time_name='intakeoutputoffset').reset_index(drop=True)

## urine_output
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_urine_output.csv'), engine='python')

for vital_name in ['urineoutput']:
    actions = get_vital(vital_name).reset_index(drop=True)
    
## med_binary
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_med_binary.csv'), engine='python')

for vital_name in ['med_sedation', 'med_neuromuscular_blocker']:
    actions = get_vital(vital_name, time_name='treatmentoffset').reset_index(drop=True)
actions = actions.fillna({'med_sedation':0, 'med_neuromuscular_blocker':0})

## action stratify low:0, medium:1, high:2
actions['PEEP_level'] = actions['PEEP'].apply(lambda x: 0 if x<5 else 1 if x==5 else 2)
actions['FiO2_level'] = actions['FiO2'].apply(lambda x: 0 if x<=35 else 1 if x<50 else 2)
actions['Tidal_level'] = actions['Tidal'].apply(lambda x: 0 if x<=6.5 else 1 if x<8 else 2)

## demography
actions = pd.merge(actions, patients[['patientunitstayid', 'age', 'gender', 'apache_iv', 'admissionweight',
                                      'sofatotal', 'hosp_mort']], on='patientunitstayid', how='left')

actions = actions.fillna({'age':63.0})

# save temp
actions.to_csv(os.path.join(output_dir,'temp_data_rl.csv'), index=False)

# state, action, next_state
state_cols = ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp',
       'lactate', 'bicarbonate', 'wbc', 'pao2', 'paco2', 'pH', 'gcs',
       'intaketotal', 'nettotal', 'urineoutput', 'med_sedation',
       'med_neuromuscular_blocker', 'age', 'gender', 'apache_iv', 'admissionweight', 'sofatotal']

# fill values
na_fill = {}
for col_name in state_cols:
    na_fill[col_name] = actions[col_name].median()
actions = actions.fillna(na_fill)
with open(os.path.join(output_dir,'na_fill.pkl'), 'wb') as f:
    pickle.dump(na_fill, f)

# normalize
origin_rename = {i:'ori_'+i for i in state_cols}
actions = actions.rename(columns=origin_rename)
ranges = {}
for col_name in state_cols:
    ranges[col_name] = [actions['ori_'+col_name].min(), actions['ori_'+col_name].max()]
    actions[col_name] = actions['ori_'+col_name].apply(lambda x: (x-ranges[col_name][0]) / (ranges[col_name][1]-ranges[col_name][0]+0.0000001))
with open(os.path.join(output_dir,'ranges.pkl'), 'wb') as f:
    pickle.dump(ranges, f)

# next state
next_rename = {i:'next_'+i for i in state_cols}
next_rename['ori_spo2'] = 'next_ori_spo2'
next_data = actions.copy().rename(columns=next_rename)
next_data['step_id'] = next_data['step_id'].apply(lambda x: x-1)
actions = pd.merge(actions, next_data[['patientunitstayid','step_id']+list(next_rename.values())], on=['patientunitstayid','step_id'], how='left')
actions['done'] = actions['next_age'].apply(lambda x: 1 if pd.isnull(x) else 0)

# save data
actions.to_csv(os.path.join(output_dir,'data_rl.csv'), index=False)

## plot level
def showonepatientlevel(pid):                                                                                                            
    peep = {}
    fio2 = {}
    tidal_ten = {}
    p1 = actions[actions['patientunitstayid']==pid]
    for i in p1.index:
        if not pd.isnull(p1.loc[i,'PEEP']):
            peep[p1.loc[i,'time']/240] = p1.loc[i,'PEEP_level']
        if not pd.isnull(p1.loc[i,'FiO2']):
            fio2[p1.loc[i,'time']/240] = p1.loc[i,'FiO2_level']+3
        if not pd.isnull(p1.loc[i,'Tidal']):
            tidal_ten[p1.loc[i,'time']/240] = p1.loc[i,'Tidal_level']+6
    fig, ax = plt.subplots(figsize=(30,5))
    ax.scatter(peep.keys(), peep.values(), color='green', label='PEEP')
    ax.scatter(fio2.keys(), fio2.values(), color='red', label='FiO2')
    ax.scatter(tidal_ten.keys(), tidal_ten.values(), color='blue', label='Tidal')
    ax.set_xlabel('time (4h)')
    ax.legend()
    ax.grid(axis='x')
#showonepatientlevel(141304)