#!/usr/bin/env python
# coding: utf-8

# In[330]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
#import seaborn as sns


# In[331]:


def showvalues(values):
    print('missing_rate: %f\nmedian: %.2f, max: %.2f, min: %.2f' % (values.isnull().sum()/values.shape[0], values.median(), values.max(), values.min()))
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(values[~pd.isnull(values)], vert=False, showfliers=False)
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(values[~pd.isnull(values)], vert=False, showfliers=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(values[~pd.isnull(values)])


# In[454]:


data_dir = '../mimic_data_v1'
output_dir = '../mimic_v2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[455]:


## patients
patients = pd.read_csv(os.path.join(data_dir,'cohort.csv'), engine='python')
demo = pd.read_csv(os.path.join(data_dir,'demographic.csv'), engine='python')
patients = pd.merge(patients, demo, on='patientunitstayid', how='left')
patients['vent_time'] = patients['vent_end']-patients['vent_start']

# age, exclude >89 and < 16, from 54728 to 52715
patients = patients[patients['age'].apply(lambda x: (False if pd.isnull(x) else False if int(x)<16 else True) if '>' not in str(x) else False)]
patients['age'] = patients['age'].apply(lambda x: int(x))


# In[456]:


patients.shape


# In[334]:


patients.head()


# In[457]:


# height, exclude >=250 or <=130, from 52715 to ?
#showvalues(patients['admissionheight'])
patients = patients[patients['admissionheight'].apply(lambda x: True if pd.isnull(x) else x>130 and x<250)]
patients = patients[patients['admissionheight'].apply(lambda x: not pd.isnull(x))]


# In[458]:


patients.shape


# In[459]:


# gender, # 0 female, 1 male, exclude null, from ? to 52092
#print(patients.gender.value_counts())
#values = patients['gender']
#print('missing_rate: %f\nmedian: %.2f, max: %.2f, min: %.2f' % (values.isnull().sum()/values.shape[0], values.median(), values.max(), values.min()))
patients = patients[patients['gender'].apply(lambda x: not pd.isnull(x))]


# In[460]:


patients.shape


# In[461]:


patients['gender'] = patients.apply(lambda x: 0 if x['gender']=='F' else 1, axis=1)
patients.gender = patients.gender.astype(np.float32)


# In[462]:


# Ideal body weight is computed in men as 50 + (0.91 × [height in centimeters − 152.4]) and in women as 45.5 + (0.91 × [height in centimeters − 152.4])
def idealweight(x):
    if x['gender']==1: #male
        result = 50 + 0.91 * (x['admissionheight']-152.4)
    else:
        result = 45.5 + 0.91 * (x['admissionheight']-152.4)
    return result
patients['idealweight'] = patients.apply(idealweight, axis=1)


# In[463]:


# exclude too long vent_time (>20000), remain 48424
patients = patients[patients['vent_time']<=20000]

print('patients remaining', patients.shape[0])


# In[464]:


## actions
actions = pd.read_csv(os.path.join(data_dir,'pivot_action.csv'), engine='python')
actions.head()


# In[465]:


actions['PEEP'] = actions['PEEP_1'].apply(lambda x: float(x) if x!='100%' else np.nan)


# In[466]:


actions['FiO2'] = actions['FiO2_1']
actions['FiO2'] = actions['FiO2'].apply(lambda x: np.nan if pd.isnull(x) else float(x) if '%' not in str(x) else float(x.replace('%','')))

actions['Tidal_volume'] = actions.apply(lambda x: (x['Tidal_volume_3'] if pd.isnull(x['Tidal_volume_2']) else x['Tidal_volume_2']) if pd.isnull(x['Tidal_volume_1']) else x['Tidal_volume_1'] , axis=1)
actions['Tidal_volume'] = actions['Tidal_volume'].apply(lambda x: float(x) if x!='400%' else np.nan)

print('action rows', actions.shape[0])


# In[467]:


## actions, rows 2593399
actions = pd.merge(actions, patients, on='patientunitstayid', how='left')
# exclude bad patient info, remain 1511518, 32692 patients
actions = actions[~pd.isnull(actions['vent_start'])]


# In[468]:


# exclude actions out of vent time. remain 1438738, 32040 patients
actions = actions[actions.apply(lambda x: x['chartoffset']>=x['vent_start'] and x['chartoffset']<=x['vent_end'], axis=1)]


# In[469]:


# Tidal to ml/kg ideal weight
actions['Tidal'] = actions.apply(lambda x: np.nan if pd.isnull(x['Tidal_volume']) else x['Tidal_volume']/x['idealweight'], axis=1)


# In[470]:

actions['PEEP'] = actions['PEEP_1']


# In[474]:


actions.tail(2)


# In[475]:


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


# In[476]:


# split actions into several 4h
def get_actions(action_type):
    temp = actions[~pd.isnull(actions[action_type])][['patientunitstayid','chartoffset',action_type,'vent_start','vent_end']].reset_index(drop=True)
    temp = temp.sort_values(by=['patientunitstayid','chartoffset'])
    def get_action_result(data):
        # 4h (240 minutes) per step
        result = pd.DataFrame(list(range(int(data.vent_start.values[0])+120, int(data.vent_end.values[0]), 240)),columns=['time'])
        result['patientunitstayid'] = data['patientunitstayid'].values[0]
        result = pd.merge(result, data[['patientunitstayid','chartoffset',action_type]], on='patientunitstayid', how='outer')
        result['time_offset'] = abs(result['time']-result['chartoffset'])
        result = result[result['time_offset']<=120]
        result = result.groupby(['patientunitstayid','time']).mean().reset_index().drop(['chartoffset','time_offset'],axis=1)
        return result
    new_actions = temp.groupby('patientunitstayid').apply(get_action_result)
    return new_actions


# In[477]:


peep_actions = get_actions('PEEP').drop(['patientunitstayid'],axis=1).reset_index().rename(columns={'level_1':'step_id'})
fio2_actions = get_actions('FiO2').drop(['patientunitstayid'],axis=1).reset_index().rename(columns={'level_1':'step_id'})
tidal_actions = get_actions('Tidal').drop(['patientunitstayid'],axis=1).reset_index().rename(columns={'level_1':'step_id'})

old_actions = actions.copy()


# In[358]:


del actions
def time_actions(data):
    result = pd.DataFrame(list(range(int(data.vent_start.values[0])+120, int(data.vent_end.values[0]), 240)),columns=['time'])
    result['patientunitstayid'] = data['patientunitstayid'].values[0]
    return result


# In[359]:


actions = old_actions.groupby('patientunitstayid').apply(time_actions)
actions = actions.drop(['patientunitstayid'],axis=1).reset_index().rename(columns={'level_1':'step_id'})

actions = pd.merge(actions, peep_actions, on=['patientunitstayid','step_id','time'], how='left')
actions = pd.merge(actions, fio2_actions, on=['patientunitstayid','step_id','time'], how='left')
actions = pd.merge(actions, tidal_actions, on=['patientunitstayid','step_id','time'], how='left')


# In[360]:


# action missing rate
value_missing = actions.groupby('patientunitstayid').apply(lambda x: x[x.apply(lambda a: pd.isnull(a['PEEP']) and pd.isnull(a['FiO2']) and pd.isnull(a['Tidal']), axis=1)].shape[0]/x.shape[0])
value_missing = value_missing.reset_index().rename(columns={0:'action_missing_rate'}) # median 0.89
value_missing.to_csv(os.path.join(output_dir,'action_missing_rate.csv'), index=False)
#TODO should be <0.15 here
#value_missing = value_missing[value_missing['action_missing_rate']<0.15]
#actions = actions[actions['patientunitstayid'].isin(value_missing.patientunitstayid)]


# In[453]:


value_missing.shape


# In[361]:


plt.hist(value_missing.action_missing_rate)


# In[362]:


#TODO too much fills
# use last to fill na
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

for i in range(actions.shape[0]-2,-1,-1):
    if actions.loc[i,'patientunitstayid']==actions.loc[i+1,'patientunitstayid']:
        if pd.isnull(actions.loc[i,'PEEP']):
            actions.loc[i,'PEEP'] = actions.loc[i+1,'PEEP']
        if pd.isnull(actions.loc[i,'FiO2']):
            actions.loc[i,'FiO2'] = actions.loc[i+1,'FiO2']
        if pd.isnull(actions.loc[i,'Tidal']):
            actions.loc[i,'Tidal'] = actions.loc[i+1,'Tidal']

actions = actions[~actions.apply(lambda x: pd.isnull(x['PEEP']) or pd.isnull(x['FiO2']) or pd.isnull(x['Tidal']), axis=1)]

# save temperory result of actions
actions.to_csv(os.path.join(output_dir,'actions_4h.csv'), index=False)
patients.to_csv(os.path.join(output_dir,'remain_patients.csv'), index=False)

print('action rows remove too much missing', actions.shape[0])
print('action patients number after above', actions.patientunitstayid.unique().shape)


# In[363]:


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


# In[365]:


showonepatient(39997976)


# In[367]:


## vital signs
vital = pd.read_csv(os.path.join(data_dir, 'pivot_vital.csv'), engine='python')

vital['sbp'] = vital.apply(lambda x: x['nibp_systolic'] if pd.isnull(x['ibp_systolic']) else x['ibp_systolic'], axis=1)
vital['dbp'] = vital.apply(lambda x: x['nibp_distolic'] if pd.isnull(x['ibp_diastolic']) else x['ibp_diastolic'], axis=1)
vital['mbp'] = vital.apply(lambda x: x['nibp_mean'] if pd.isnull(x['ibp_mean']) else x['ibp_mean'], axis=1)
vital = vital.drop(['nibp_systolic', 'nibp_distolic', 'nibp_mean', 'ibp_systolic', 'ibp_diastolic', 'ibp_mean'], axis=1)


# In[368]:


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


# In[369]:


for vital_name in ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp','mbp']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[370]:


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
    


# In[371]:


## fluid
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_fluid.csv'), engine='python')
#TODO intaketotal seems not cumulative value
vital = vital.sort_values(by=['patientunitstayid', 'chartoffset'])

vital = vital.groupby('patientunitstayid').apply(lambda x: x.reset_index(drop=True).reset_index().rename(columns={'index':'time_id'})).reset_index(drop=True)
vital_next = vital.copy().rename(columns={'intake_total':'last_intake_total','output_total':'last_output_total','nettotal':'last_nettotal','chartoffset':'last_chartoffset'})
vital_next['time_id'] = vital['time_id']+1
vital = pd.merge(vital, vital_next, on=['patientunitstayid','time_id'], how='left')
vital['intake_hours'] = vital.apply(lambda x: x['intake_total']-x['last_intake_total'] if not pd.isnull(x['last_intake_total']) else x['intake_total'], axis=1)
vital['output_hours'] = vital.apply(lambda x: x['output_total']-x['last_output_total'] if not pd.isnull(x['last_output_total']) else x['output_total'], axis=1)

for vital_name in ['intake_hours','output_hours']:
    actions = get_vital(vital_name, time_name='chartoffset', mode='sum').reset_index(drop=True)
    actions = actions.fillna({vital_name:0})


# In[375]:


## med_binary
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_med_binary.csv'), engine='python')

for vital_name in ['med_sedation', 'med_neuromuscular_blocker']:
    actions = get_vital(vital_name, time_name='chartoffset').reset_index(drop=True)
actions = actions.fillna({'med_sedation':0, 'med_neuromuscular_blocker':0})

# In[376]:


# ## action stratify low:0, medium:1, high:2
# actions['PEEP_level'] = actions['PEEP'].apply(lambda x: 0 if x<5 else 1 if x==5 else 2)
# actions['FiO2_level'] = actions['FiO2'].apply(lambda x: 0 if x<=35 else 1 if x<50 else 2)
# actions['Tidal_level'] = actions['Tidal'].apply(lambda x: 0 if x<=6.5 else 1 if x<8 else 2)

# ## demography
# actions = pd.merge(actions, patients[['patientunitstayid', 'age', 'gender', 'admissionweight',
#                                       'sofatotal', 'hosp_mort']], on='patientunitstayid', how='left')

# actions = actions.fillna({'age':63.0})

## action stratify low:0, medium:1, high:2
actions['PEEP_level'] = actions['PEEP'].apply(lambda x: 0 if x<=5 else 1)
actions['FiO2_level'] = actions['FiO2'].apply(lambda x: 0 if x<=35 else 1 if x<50 else 2)
actions['Tidal_level'] = actions['Tidal'].apply(lambda x: 0 if x<=6.5 else 1 if x<8 else 2)

## demography
actions = pd.merge(actions, patients[['patientunitstayid', 'age', 'gender', 'admissionweight',
                                      'sofatotal', 'hosp_mort']], on='patientunitstayid', how='left')

actions = actions.fillna({'age':63.0})


# In[426]:

# save temp
actions.to_csv(os.path.join(output_dir,'temp_data_rl.csv'), index=False)

# In[377]:
vital = pd.read_csv(os.path.join(data_dir, 'pivot_vasopressor_edit.csv'), engine='python')
vital = vital[vital['equivalent_mcg_kg_min'].apply(lambda x: x!='na')]
vital['equivalent_mcg_kg_min'] = vital['equivalent_mcg_kg_min'].apply(lambda x: float(x))

vital = vital[vital['Duration_min'].apply(lambda x: x!='na')]
vital['Duration_min'] = vital['Duration_min'].apply(lambda x: float(x))


# dose each 4h
old_vital = vital[['patientunitstayid','chartoffset','equivalent_mcg_kg_min','Duration_min']].reset_index(drop=True)
old_vital = old_vital.sort_values(by=['patientunitstayid','chartoffset'])
old_vital = old_vital.reset_index()


def get_vasopressor_result(data):
    # 2h (240 minutes) per step, this will not miss
    result = pd.DataFrame(list(range(int(data['chartoffset'].values[0]), int(data['Duration_min'].values[0]+data['chartoffset'].values[0]), 120)),columns=['drug_time'])
    result['patientunitstayid'] = data['patientunitstayid'].values[0]
    result['equivalent_mcg_kg_min'] = data['equivalent_mcg_kg_min'].values[0]
    result['Duration_min'] = data['Duration_min'].values[0]
    return result


vital = old_vital.groupby('index').apply(get_vasopressor_result)
vital = vital.sort_values(by=['patientunitstayid','drug_time']).reset_index(drop=True)
vital.head(5)


# In[426]:

# use values between time-120 and time+120. 
def get_dose_vital(vital_name, time_name='chartoffset', mode='sum'):
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
            result = result.groupby(['patientunitstayid','time']).sum().reset_index()
        return result
    new_vital = actions[['patientunitstayid','time']].groupby('patientunitstayid').apply(get_vital_result).reset_index(drop=True)
    new_actions = pd.merge(actions, new_vital, on=['patientunitstayid','time'], how='left')
    return new_actions


# In[427]:

actions = get_dose_vital('equivalent_mcg_kg_min', time_name='drug_time', mode='sum').reset_index(drop=True)
actions['equivalent_mg_4h'] = actions.apply(lambda x: x['equivalent_mcg_kg_min']*24, axis=1)
actions = actions.drop(['equivalent_mcg_kg_min'], axis=1)


# In[408]:


# state, action, next_state
state_cols = ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp', 'mbp',
       'lactate', 'bicarbonate', 'wbc', 'pao2', 'paco2', 'pH', 'gcs',
       'intaketotal', 'urineoutput', 'med_sedation',
       'med_neuromuscular_blocker', 'age', 'gender','admissionweight', 'sofatotal',
       'equivalent_mg_4h']


# fill values
na_fill = {}
for col_name in state_cols:
    na_fill[col_name] = actions[col_name].median()
actions = actions.fillna(na_fill)
with open(os.path.join(output_dir,'mimic_na_fill.pkl'), 'wb') as f:
    pickle.dump(na_fill, f)


# In[410]:


# normalize
origin_rename = {i:'ori_'+i for i in state_cols}
actions = actions.rename(columns=origin_rename)
ranges = {}
log_columns = ['intaketotal','urineoutput','lactate','equivalent_mg_4h']
for col_name in state_cols:
    if col_name in log_columns:
        ranges[col_name] = [np.log(actions['ori_'+col_name]+1).min(), np.log(actions['ori_'+col_name]+1).max()]
        actions[col_name] = actions['ori_'+col_name].apply(lambda x: (np.log(x+1)-ranges[col_name][0]) / (ranges[col_name][1]-ranges[col_name][0]+0.0000001))
    else:
        ranges[col_name] = [actions['ori_'+col_name].min(), actions['ori_'+col_name].max()]
        actions[col_name] = actions['ori_'+col_name].apply(lambda x: (x-ranges[col_name][0]) / (ranges[col_name][1]-ranges[col_name][0]+0.0000001))
with open(os.path.join(output_dir,'mimic_ranges.pkl'), 'wb') as f:
    pickle.dump(ranges, f)


# In[411]:


# next state
next_rename = {i:'next_'+i for i in state_cols}
next_rename['ori_spo2'] = 'next_ori_spo2'
next_rename['ori_mbp'] = 'next_ori_mbp'
next_data = actions.copy().rename(columns=next_rename)
next_data['step_id'] = next_data['step_id'].apply(lambda x: x-1)
actions = pd.merge(actions, next_data[['patientunitstayid','step_id']+list(next_rename.values())], on=['patientunitstayid','step_id'], how='left')
actions['done'] = actions['next_age'].apply(lambda x: 1 if pd.isnull(x) else 0)


# In[412]:


# save data
actions.to_csv(os.path.join(output_dir,'data_rl.csv'), index=False)


# In[413]:


actions.columns


# In[414]:


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


# In[415]:


showonepatientlevel(39997976)


# In[416]:


actions.shape


# # Vasopressor

# In[417]:


def showvalues(values):
    print('missing_rate: %f\nmedian: %.2f, max: %.2f, min: %.2f' % (values.isnull().sum()/values.shape[0], values.median(), values.max(), values.min()))
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(values[~pd.isnull(values)], vert=False, showfliers=False)
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(values[~pd.isnull(values)], vert=False, showfliers=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(values[~pd.isnull(values)])


# In[431]:


actions.head(10)


# In[432]:


actions.columns


# In[281]:


# origin_rename = {'ori_output_total':'ori_urineoutput', 'ori_Gender':'ori_gender',
#                  'next_output_total':'next_urineoutput','next_Gender':'next_gender',
#                  'output_total':'urineoutput','Gender':'gender',
#                  'ori_intake_total':'ori_intaketotal', 'intake_total': 'intaketotal', 'next_intake_total':'next_intaketotal'
#                 }
# actions = actions.rename(columns=origin_rename)



# In[322]:


# np.median(test['sofatotal'])
# np.max(test['sofatotal'])
# np.min(test['sofatotal'])


# In[438]:


# df = pd.read_csv(os.path.join('../eICU', 'data_rl_with_dose.csv'), engine='python')


# In[446]:


# df.columns


# In[447]:


# df1 = df[df['step_id']==0]


# In[448]:


# df1.shape


# In[449]:


# df1[df1['hosp_mort']==1].shape


# In[450]:


# 1202/4802*100


# In[ ]:




