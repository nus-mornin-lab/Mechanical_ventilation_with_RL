import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

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

actions = pd.read_csv(os.path.join(output_dir,'actions_4h.csv'), engine='python')
patients = pd.read_csv(os.path.join(output_dir,'remain_patients.csv'), engine='python')

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

for vital_name in ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp', 'mbp']:
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

for vital_name in ['intaketotal']:
    actions = get_vital(vital_name, time_name='intakeoutputoffset', mode='sum').reset_index(drop=True)

## urine_output
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_urine_output.csv'), engine='python')

for vital_name in ['urineoutput']:
    actions = get_vital(vital_name, mode='sum').reset_index(drop=True)
    
abnormals = {}
abnormals['intaketotal'] = [0, 13000]
abnormals['urineoutput'] = [0, 8200]
def remove_abnormal(data, col):
    data[col] = data[col].apply(lambda x: np.nan if x<abnormals[col][0] or x>abnormals[col][1] else x)
remove_abnormal(actions, 'intaketotal')
remove_abnormal(actions, 'urineoutput')

actions = actions.fillna({'intaketotal':0, 'urineoutput':0})

## med_binary
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_med_binary.csv'), engine='python')

for vital_name in ['med_sedation', 'med_neuromuscular_blocker']:
    actions = get_vital(vital_name, time_name='treatmentoffset').reset_index(drop=True)
actions = actions.fillna({'med_sedation':0, 'med_neuromuscular_blocker':0})

## action stratify low:0, medium:1, high:2, or low:0, high:1
#TODO PEEP level already changed
actions['PEEP_level'] = actions['PEEP'].apply(lambda x: 0 if x<=5 else 1)
actions['FiO2_level'] = actions['FiO2'].apply(lambda x: 0 if x<=35 else 1 if x<50 else 2)
actions['Tidal_level'] = actions['Tidal'].apply(lambda x: 0 if x<=6.5 else 1 if x<8 else 2)

## demography
actions = pd.merge(actions, patients[['patientunitstayid', 'age', 'gender', 'admissionweight',
                                      'sofatotal', 'hosp_mort']], on='patientunitstayid', how='left')

actions = actions.fillna({'age':63.0})

# save temp
actions.to_csv(os.path.join(output_dir,'temp_data_rl.csv'), index=False)

## dose
# use values between time-120 and time+120. 
def get_dose_vital(vital_name, time_name='chartoffset', mode='nearest'):
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
    new_actions = pd.merge(actions, new_vital, on=['patientunitstayid','time'], how='left')
    return new_actions

## vasopressor_edit
del vital
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

actions = get_dose_vital('equivalent_mg_4h', time_name='drug_time', mode='mean').reset_index(drop=True)
# fill na
actions = actions.fillna({'equivalent_mg_4h':0})

## state, action, next_state
#TODO 去掉了nettotal
state_cols = ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp', 'mbp',
       'lactate', 'bicarbonate', 'wbc', 'pao2', 'paco2', 'pH', 'gcs',
       'intaketotal', 'urineoutput', 'med_sedation', 
       'med_neuromuscular_blocker', 'age', 'gender', 'admissionweight', 'sofatotal',
       'equivalent_mg_4h']

# fill values
# use mimic na fill
with open(os.path.join(output_dir,'mimic_na_fill.pkl'), 'rb') as f:
    na_fill = pickle.load(f)
for col_name in state_cols:
    na_fill[col_name] = actions[col_name].median()
actions = actions.fillna(na_fill)

# normalize
origin_rename = {i:'ori_'+i for i in state_cols}
actions = actions.rename(columns=origin_rename)
# use mimic range
with open(os.path.join(output_dir,'mimic_ranges.pkl'), 'rb') as f:
    ranges = pickle.load(f)
log_columns = ['intaketotal','urineoutput','lactate','equivalent_mg_4h']
for col_name in state_cols:
    if col_name in log_columns:
        actions[col_name] = actions['ori_'+col_name].apply(lambda x: (np.log(x+1)-ranges[col_name][0]) / (ranges[col_name][1]-ranges[col_name][0]+0.0000001))
    else:
        actions[col_name] = actions['ori_'+col_name].apply(lambda x: (x-ranges[col_name][0]) / (ranges[col_name][1]-ranges[col_name][0]+0.0000001))

# next state
next_rename = {i:'next_'+i for i in state_cols}
next_rename['ori_spo2'] = 'next_ori_spo2'
next_rename['ori_mbp'] = 'next_ori_mbp'
next_data = actions.copy().rename(columns=next_rename)
next_data['step_id'] = next_data['step_id'].apply(lambda x: x-1)
actions = pd.merge(actions, next_data[['patientunitstayid','step_id']+list(next_rename.values())], on=['patientunitstayid','step_id'], how='left')
actions['done'] = actions['next_age'].apply(lambda x: 1 if pd.isnull(x) else 0)

# save data
actions.to_csv(os.path.join(output_dir,'data_rl.csv'), index=False)
