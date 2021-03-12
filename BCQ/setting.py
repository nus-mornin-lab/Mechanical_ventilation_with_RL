# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:04:52 2020

@author: 18795
"""
import numpy as np
import pickle
state_col = pickle.load(open('data/state_columns_mimic.pkl','rb'))
next_state_col = ['next_' + f for f in state_col]

# no apache now
# state_col = ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp', 'lactate',
#              'bicarbonate', 'wbc', 'pao2', 'paco2', 'pH', 'gcs', 'intaketotal',
#              'urineoutput', 'med_sedation', 'med_neuromuscular_blocker', 'age', 'gender',
#              'admissionweight', 'sofatotal', 'equivalent_mg_4h']
# next_state_col = ['next_heartrate', 'next_respiratoryrate', 'next_spo2', 'next_temperature',
#                   'next_sbp', 'next_dbp', 'next_lactate', 'next_bicarbonate', 'next_wbc',
#                   'next_pao2', 'next_paco2', 'next_pH', 'next_gcs', 'next_intaketotal',
#                    'next_urineoutput', 'next_med_sedation', 'next_med_neuromuscular_blocker',
#                   'next_age', 'next_gender', 'next_admissionweight', 'next_sofatotal', 'next_equivalent_mg_4h']
action_dis_col = ['PEEP_level', 'FiO2_level', 'Tidal_level']
# reward_col = ['ori_spo2', 'next_ori_spo2', 'hosp_mort']

SEED = 7
ITERATION_ROUND = 10
ACTION_SPACE = 18# 27
BATCH_SIZE = 256
GAMMA = 0.99
VALIDATION = False

MODEL = 'BCQ' # 'DQN' 'FQI'

REWARD_FUN = 'reward_short_long_usd_new'

def reward_short_long_usd2(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 1
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 1
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 1
    else:
        res = np.nan
    return res

def reward_short_long_usd(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 2
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 2
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 1
    else:
        res = np.nan
    return res

def reward_short_long10_2(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -5
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 2
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 1
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 2
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 1
    else:
        res = np.nan
    return res

def reward_short_long10(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 2
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 2
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 2
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 2
    else:
        res = np.nan
    return res

def reward_only_long_positive10(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = 0
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10
    elif x['done'] == 0:
        res = 0
    else:
        res = np.nan
    return res

def reward_only_long(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -1
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 1
    elif x['done'] == 0:
        res = 0
    else:
        res = np.nan
    return res

def reward_only_long_positive(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = 0
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 1
    elif x['done'] == 0:
        res = 0
    else:
        res = np.nan
    return res

def reward_short_long_spo2(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res += -1
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res += 1
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.1
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.1
    else:
        res = np.nan
    return res

def reward_short_long_spo2_positive(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res += 0
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res += 1
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.1
    else:
        res = np.nan
    return res

def reward_short_long_usd_new(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.5
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.5
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.25
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.25
    else:
        res = np.nan
    return res

'''
def reward_short_long_spo2_mbp_positive(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res += 0
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res += 1
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.1
    else:
        res = np.nan
    return res
'''
