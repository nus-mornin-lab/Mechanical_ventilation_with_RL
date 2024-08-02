
import numpy as np
import pickle

SEED = 523                                 # 可调  主要影响参数初始化和训练时选择数据的顺序
ITERATION_ROUND = 30                     # 可调  大约遍历数据集多少次
ACTION_SPACE = 18# 27
BATCH_SIZE = 256
GAMMA = 0.99                             # 可调  discount factor γ, 建议0.9 ~ 0.995之间
VALIDATION = False                       # 是否使用验证集自适应选择超参数

DATA_DATE = '0420'   # 0326                数据版本
TIME_RANGE = '24h'   # 24h                 入组条件，目前使用>=24h条件入组的患者
CUT_TIME = '72h'   # 48h/72h/14d           可调  截断时间，episode长度
TRAIN_SET = 'mimic'   # mimic/eicu         可调  训练数据集，在mimic/eicu上训练 （在另一个数据集上测试）
STEP_LENGTH = '240min'   # 240min/60min    可调  每个time_step的长度
CRITICAL_STATE = False   # True/False       可调  是否仅使用关键state

REWARD_FUN = 'reward_short_long_usd'     # 可调  在本py中定义新的reward函数

MODEL = 'DQN' 

if CRITICAL_STATE:
    state_col = pickle.load(open('../data/%s/state_columns_critical.pkl'%(DATA_DATE),'rb'))
else:
    state_col = pickle.load(open('../data/%s/state_columns.pkl'%(DATA_DATE),'rb'))
    
next_state_col = ['next_' + f for f in state_col]

action_dis_col = ['PEEP_level', 'FiO2_level', 'Tidal_level']
other_related_col = ['ori_sofa_24hours','ori_spo2','ori_mbp']
other_related_next_col = ['next_ori_sofa_24hours','next_ori_spo2','next_ori_mbp'] # 需和上面一个对应

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

    return res

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

    return res

def reward_only_long_positive10(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = 0
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10
    elif x['done'] == 0:
        res = 0

    return res

def reward_only_long(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -1
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 1
    elif x['done'] == 0:
        res = 0

    return res

def reward_only_long_positive(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = 0
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 1
    elif x['done'] == 0:
        res = 0

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

    return res

def reward_short_long_usd_new2(x):
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
            res += 0.5
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.5

    return res

def reward_short_long_usd_new3(x):
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

    return res

def reward_short_long_usd_new4(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10


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

    return res
'''

def reward_short_long_usd_new_plus(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -5
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 10
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.5
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.25
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0

    return res

'''
state_col = pickle.load(open('../data/%s/state_columns.pkl'%(DATA_DATE),'rb'))
not_critical_states = [
'chloride', 'potassium', 'sodium', 'glucose', 'hemoglobin', 'bun', 'creatinine',
'albumin', 'magnesium', 'calcium', 'ionized_calcium', 'platelet', 'pt', 'ptt',
'inr', 'bilirubin_total']
critical_states = list(set(state_col).difference(set(not_critical_states)))
with open('../data/%s/state_columns_not_critical.pkl'%(DATA_DATE),'wb') as fw:
    pickle.dump(not_critical_states,fw)
with open('../data/%s/state_columns_critical.pkl'%(DATA_DATE),'wb') as fw:
    pickle.dump(critical_states,fw)
'''