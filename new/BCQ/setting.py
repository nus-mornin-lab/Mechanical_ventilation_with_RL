
import numpy as np
import pickle
import pandas as pd

SEED = 777                                 # 主要影响参数初始化和训练时选择数据的顺序
ITERATION_ROUND = 21                     # round 大约遍历数据集多少次
ACTION_SPACE = 18# 27
BATCH_SIZE = 256
GAMMA = 0.99                             # discount factor γ, 建议0.9 ~ 0.995之间
VALIDATION = False                       # 是否使用验证集自适应选择超参数
ROUND_PER_EVAL = 1                       # 多少个round评估一次, 每个round代表约遍历一次训练数据

DATA_DATE = '0523'   # 0326                数据版本
TIME_RANGE = '24h'   # 24h                 入组条件，目前使用>=24h条件入组的患者
CUT_TIME = '48h'   # 24h/48h/72h/7d/14d    截断时间，episode长度
TRAIN_SET = 'eicu'   # mimic/eicu         训练数据集，在mimic/eicu上训练 （在另一个数据集上测试）
STEP_LENGTH = '240min'   # 240min/60min    每个time_step的长度
CRITICAL_STATE = False   # True/False       是否仅使用关键state
TOP_STATE = None  # None/ 3-10            用随机森林选出的top n的特征来作为state
MISSING_CUT = False  # True/False           是否只用missing1为0的数据
PRETRAIN = True
RE_DIVIDE_SET = False  #                   是否根据seed重新划分训练/测试数据集

REWARD_FUN = 'reward_short_long_defined'     # 在本py中定义新的reward函数

setting_paras = {}
setting_paras['TRAIN_SET'] = ['mimic','eicu'] # change here for grid search
# setting_paras['CUT_TIME'] = ['24h','48h','72h'] # change here for grid search
# setting_paras['STEP_LENGTH'] = ['240min','60min'] # change here for grid search
setting_paras['MISSING_CUT'] = [False,True] # change here for grid search
setting_paras['SEED'] = [777,523,666] # change here for grid search
setting_paras['a'] = [20,10] # reward # change here for grid search
setting_paras['b'] = [1,2,4] # reward # change here for grid search
setting_paras['c'] = [1,2,4] # reward # change here for grid search
# setting_paras['BCQ_THRESHOLD'] = [0, 0.1,0.2,0.3] # change here for grid search
# setting_paras['GAMMA'] = [0.99] # change here for grid search 0.99/0.95/0.9
# setting_paras['PRETRAIN'] = [True, False] # change here for grid search



MODEL = 'BCQ' 

if CRITICAL_STATE:
    state_col = pickle.load(open('../data/%s/state_columns_critical.pkl'%(DATA_DATE),'rb'))
else:
    state_col = pickle.load(open('../data/%s/state_columns.pkl'%(DATA_DATE),'rb'))
    
if TOP_STATE != None:
    state_dt = pd.read_excel('../data/%s/State_features.xlsx'%(DATA_DATE))
    state_col = state_dt['Top%s'%(str(TOP_STATE))].dropna().tolist()
    
next_state_col = ['next_' + f for f in state_col]

action_dis_col = ['PEEP_level', 'FiO2_level', 'Tidal_level']
other_related_col = ['ori_sofa_24hours','ori_spo2','ori_mbp']
other_related_next_col = ['next_ori_sofa_24hours','next_ori_spo2','next_ori_mbp'] # 需和上面一个对应

def reward_short_long_defined(x, a=20, b=2, c=2):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -a/2
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = a
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += b
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= b/2
        if (x['ori_mbp'] < 65 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 65 and x['next_ori_mbp'] <= 80):
            res += c
        elif (x['ori_mbp'] >= 65 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <65 or x['next_ori_mbp'] > 80):
            res -= c/2

    return res

def reward_short_long_usd_H3_mod9(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 1
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.5
        if (x['ori_mbp'] < 65 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 65 and x['next_ori_mbp'] <= 80):
            res += 2
        elif (x['ori_mbp'] >= 65 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <65 or x['next_ori_mbp'] > 80):
            res -= 1

    return res

def reward_short_long_usd_H3_mod8(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    return res

def reward_short_long_usd_H3_mod7(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -4
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 1
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.2
        if (x['ori_mbp'] < 65 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 65 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 65 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <65 or x['next_ori_mbp'] > 80):
            res -= 0.2

    return res

def reward_short_long_usd_H3_mod6(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -4
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.5
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.2
        if (x['ori_mbp'] < 65 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 65 and x['next_ori_mbp'] <= 80):
            res += 0.5
        elif (x['ori_mbp'] >= 65 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <65 or x['next_ori_mbp'] > 80):
            res -= 0.2

    return res

def reward_short_long_usd_H3_mod5(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -5
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.5
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.25
        if (x['ori_mbp'] < 65 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 65 and x['next_ori_mbp'] <= 80):
            res += 0.5
        elif (x['ori_mbp'] >= 65 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <65 or x['next_ori_mbp'] > 80):
            res -= 0.25

    return res


def reward_short_long_usd_H3_mod4(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.5
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.25
        if (x['ori_mbp'] < 65 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 65 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 65 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <65 or x['next_ori_mbp'] > 80):
            res -= 0.5

    return res

def reward_short_long_usd_H3_mod3(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 92 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 92 and x['next_ori_spo2'] <= 98):
            res += 1
        elif (x['ori_spo2'] >= 92 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 92 or x['next_ori_spo2'] > 98):
            res -= 0.5
        if (x['ori_mbp'] < 65 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 65 and x['next_ori_mbp'] <= 80):
            res += 2
        elif (x['ori_mbp'] >= 65 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <65 or x['next_ori_mbp'] > 80):
            res -= 1

    return res

def reward_short_long_usd_H3_mod2(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 92 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 92 and x['next_ori_spo2'] <= 98):
            res += 1
        elif (x['ori_spo2'] >= 92 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 92 or x['next_ori_spo2'] > 98):
            res -= 0.5
        if (x['ori_mbp'] < 65 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 65 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 65 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <65 or x['next_ori_mbp'] > 80):
            res -= 0.5

    return res

def reward_short_long_usd_H3_mod1(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 92 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 92 and x['next_ori_spo2'] <= 98):
            res += 2
        elif (x['ori_spo2'] >= 92 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 92 or x['next_ori_spo2'] > 98):
            res -= 1
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.5

    return res

def reward_short_long_usd_H3(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 2
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 1
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.5

    return res

def reward_short_long_a_r_H11(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.01
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            # res -= 0.01
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 0.1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.01
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 2                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            # res -= 0.01
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 0.1         
    return res

def reward_short_long_a_r_H11(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.01
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            # res -= 0.01
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 0.1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.01
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 2                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            # res -= 0.01
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 0.1         
    return res

def reward_short_long_a_r_H10(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.01
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            # res -= 0.01
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 0.1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.01
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 2                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            # res -= 0.01
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 0.1         
    return res

def reward_short_long_a_r_H9(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.01
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            # res -= 0.01
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.01
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 2                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            # res -= 0.01
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 1         
    return res

def reward_short_long_a_r_H8(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.05
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            # res -= 0.01
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.05
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 2                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            # res -= 0.01
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 1         
    return res

def reward_short_long_a_r_H7(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.05
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.01
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.05
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 2                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.01
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 1         
    return res

def reward_short_long_a_r_H6(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.05
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.025
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.05
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 2                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.025
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 1         
    return res

def reward_short_long_a_r_H5(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -15
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 30
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.2
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.1
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.2
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 2                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.1
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 1         
    return res

def reward_short_long_a_r_H4(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.2
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.1
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.1
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 1                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.05
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 0.5         
    return res

def reward_short_long_a_r_H3(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 2
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 1
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.5
            
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 0.2
            if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98):
                res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.1
            if (x['ori_spo2'] >= 94 or x['ori_spo2'] <= 98):
                res -= 1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 0.1
            if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) :
                res += 1                
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.05
            if (x['ori_mbp'] >= 70 or x['ori_mbp'] <= 80) :
                res -= 0.5         
    return res

def reward_short_long_abs_H3(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 2
        elif (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 1
        if (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.5

    return res

def reward_short_long_usd_H4(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 1
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 0.5
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.5

    return res

def reward_short_long_usd_H3(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
    elif x['done'] == 0:
        if (x['ori_spo2'] < 94 or x['ori_spo2'] > 98) and (x['next_ori_spo2'] >= 94 and x['next_ori_spo2'] <= 98):
            res += 2
        elif (x['ori_spo2'] >= 94 and x['ori_spo2'] <= 98) and (x['next_ori_spo2'] < 94 or x['next_ori_spo2'] > 98):
            res -= 1
        if (x['ori_mbp'] < 70 or x['ori_mbp'] > 80) and (x['next_ori_mbp'] >= 70 and x['next_ori_mbp'] <= 80):
            res += 1
        elif (x['ori_mbp'] >= 70 and x['ori_mbp'] <= 80) and (x['next_ori_mbp'] <70 or x['next_ori_mbp'] > 80):
            res -= 0.5

    return res

def reward_short_long_usd_H2(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
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

def reward_short_long_usd_H(x):
    res = 0
    if (x['done'] == 1 and x['hosp_mort'] == 1):
        res = -10
    elif (x['done'] == 1 and x['hosp_mort'] == 0):
        res = 20
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