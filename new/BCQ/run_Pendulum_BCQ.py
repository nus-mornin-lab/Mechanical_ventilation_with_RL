import argparse
import copy
import importlib
import json
import os
import numpy as np
import torch
import discrete_BCQ
import utils
import setting
import pandas as pd
import evaluation_new
from sklearn.model_selection import KFold
import random
import itertools
from datetime import datetime
import pickle

SEED = setting.SEED
ITERATION_ROUND = setting.ITERATION_ROUND
ACTION_SPACE = setting.ACTION_SPACE
BATCH_SIZE = setting.BATCH_SIZE
GAMMA = setting.GAMMA
VALIDATION = setting.VALIDATION

DATA_DATE = setting.DATA_DATE
TIME_RANGE = setting.TIME_RANGE
CUT_TIME = setting.CUT_TIME 
TRAIN_SET = setting.TRAIN_SET
STEP_LENGTH = setting.STEP_LENGTH
CRITICAL_STATE = setting.CRITICAL_STATE

MODEL = setting.MODEL

state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
other_related_col = setting.other_related_col
other_related_next_col = setting.other_related_next_col
REWARD_FUN = setting.REWARD_FUN

# Trains BCQ offline
def train_BCQ(replay_buffer, num_actions, state_dim, device, parameters):
    # Initialize and load policy
    policy = discrete_BCQ.discrete_BCQ(
        parameters["is_atari"],
        num_actions,
        state_dim,
        device,
        parameters["BCQ_THRESHOLD"],
        parameters["discount"],
        parameters["optimizer"],
        {"lr":parameters["lr"]},
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"]
    )

    training_iters = 0
    
    while training_iters < parameters["MAX_TIMESTEPS"]: 
        print ('round: ' + str(training_iters))
        policy.train(replay_buffer)
        training_iters += 1
    Q_s, actions = policy.select_action_new(np.array(replay_buffer.state))
    return policy.loss ,policy, Q_s, actions

def divide_set(data_name, save_path):
    np.random.seed(523)
    random.seed(523)
    data = pd.read_csv(data_name)
    all_stay = np.array(data['patientunitstayid'].unique().tolist()) 
    val_test_stay = np.random.choice(all_stay, size=int(len(all_stay)*2/5), replace=False)
    val_stay = np.random.choice(val_test_stay, size=int(len(val_test_stay)/2), replace=False).tolist()
    test_stay = list(set(val_test_stay) ^ set(val_stay))
    train_stay = list(set(all_stay) ^ set(val_test_stay))
    pt_set = {}
    for k in train_stay:
        pt_set[k] = 'trainset'
    for k in test_stay:
        pt_set[k] = 'testset'
    for k in val_stay:
        pt_set[k] = 'valset'
    with open(save_path,'wb') as fw:
        pickle.dump(pt_set,fw)
    return pt_set

def pre_processing(data, SET):

    # tag some labels...
    data['set'] = data['patientunitstayid'].apply(lambda x: patient_set[SET][x])
    data['step_id'] = data['step_id'].astype(int)
    data['done'] = 0
    
    # first cut length (if need)
    if CUT_TIME == '72h':
        if STEP_LENGTH == '240min':
            data = data[data['step_id'] < 18]
        elif STEP_LENGTH == '60min':
            data = data[data['step_id'] < 72]
    elif CUT_TIME == '48h':
        if STEP_LENGTH == '240min':
            data = data[data['step_id'] < 12]
        elif STEP_LENGTH == '60min':
            data = data[data['step_id'] < 48]
    
    # then fill data and shift to tag next_col
    def f_b_fill_and_tag_done(dt):
        dt['done'].values[-1] = 1
        dt[state_col+other_related_col] = dt[state_col+other_related_col].fillna(method = 'ffill').fillna(method = 'bfill')
        return dt
        # first f_b_fill
    data = data.groupby(['patientunitstayid']).apply(f_b_fill_and_tag_done)
        # then shift to tag next_col
    data[next_state_col+other_related_next_col] = data[state_col+other_related_col].shift(-1)
            
    # then fill median
    data.loc[data['done'] == 1, next_state_col+other_related_next_col] = np.nan
    data[state_col+other_related_col+next_state_col+other_related_next_col] = data[state_col+other_related_col+next_state_col+other_related_next_col].apply(lambda x: x.fillna(x.median())) 
    data = data.reset_index(drop = True)

    # finally calculate reward and actions
    data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
    data['actions'] = data.apply(lambda x: int(x[action_dis_col[0]] * 9 + x[action_dis_col[1]] * 3 + x[action_dis_col[2]]), axis =1)
    # data.fillna(0, inplace=True)    
    return data

# calculate cwpdis
def cal_cpwdis(val_dt, parameters):
    val_dt['concordant'] = val_dt.apply(lambda x: (x['actions'] == x['ai_action']) +0 ,axis = 1)
    val_dt = val_dt.groupby('patientunitstayid').apply(cal_pnt)
    v_cwpdis = 0
    fcs = 0
    for t in range(1, max(val_dt['step_id'])+1):
        tmp = val_dt[val_dt['step_id'] == t-1]
        if sum(tmp['pnt']) > 0:
            v_cwpdis += parameters['discount']**t * (sum(tmp['reward']*tmp['pnt'])/sum(tmp['pnt']))
        if t == max(val_dt['step_id']):
            fcs = sum(tmp['pnt'])
        
    ess = sum(val_dt['pnt'])
    
    return v_cwpdis, ess, fcs
    
def cal_pnt(dt):
    dt['conc_cumsum'] = dt['concordant'].cumsum()
    dt['pnt'] = (dt['conc_cumsum'] == (dt['step_id'] + 1))+0
    return dt

if __name__ == "__main__":
    
    mimic_data = '../data/%s/data_rl_%s_mimic.csv'%(DATA_DATE, STEP_LENGTH) #'data/mimic_data_rl_with_dose_11Dec.csv' 
    eicu_data = '../data/%s/data_rl_%s_eicu.csv'%(DATA_DATE, STEP_LENGTH) #'data/data_rl_with_dose.csv' #
    mimic_set_path = '../data/%s/mimic_set.pkl'%(DATA_DATE)
    eicu_set_path = '../data/%s/eicu_set.pkl'%(DATA_DATE)
    data_for_train_test = {}
    if TRAIN_SET == 'mimic':
        data_for_train_test['train'] = mimic_data
        data_for_train_test['test'] = eicu_data
        TEST_SET = 'eicu'
    elif TRAIN_SET == 'eicu':
        data_for_train_test['train'] = eicu_data
        data_for_train_test['test'] = mimic_data
        TEST_SET = 'mimic'
    
    ### train val test 
    patient_set = {}
    if os.path.exists(mimic_set_path) == False:
        mimic_set = divide_set(mimic_data, mimic_set_path)
    else:
        mimic_set = pickle.load(open(mimic_set_path,'rb'))
            
    if os.path.exists(eicu_set_path) == False:
        eicu_set = divide_set(eicu_data, eicu_set_path)
    else:
        eicu_set = pickle.load(open(eicu_set_path,'rb'))
        
    patient_set['mimic'] = mimic_set
    patient_set['eicu'] = eicu_set

    data = pd.read_csv(data_for_train_test['train'])
    data = pre_processing(data, TRAIN_SET)
    #  aa = data[state_col + action_dis_col]
    MAX_TIMESTEPS = int(len(data) / BATCH_SIZE * ITERATION_ROUND)
    regular_parameters = {
    "is_atari":False,
    # Learning
    "MAX_TIMESTEPS":MAX_TIMESTEPS,
    "discount": GAMMA,  # 和其他model保持一致
    "BCQ_THRESHOLD": 0.3,
    # "buffer_size": MEMORY_SIZE,   
    "batch_size": BATCH_SIZE,
    "optimizer": "Adam",
    "lr": 1e-3,
    "polyak_target_update": True,  # 软/硬更新
    "target_update_freq": 50, 
    "tau": 0.01
    }
    
    # Make env and determine properties
    state_dim = len(state_col)
    num_actions = ACTION_SPACE
    parameters = regular_parameters
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # cut train val test
    train_data = data[data['set'] == 'trainset']
    train_data = train_data.reset_index(drop = True)
    val_data = data[data['set'] == 'valset']
    val_data = val_data.reset_index(drop = True)
    test_data = data[data['set'] == 'testset']
    test_data = test_data.reset_index(drop = True)
    
    if VALIDATION == True:
        # candidate parameters
        val_paras = {}
        # val_paras['discount'] = np.arange(0.7, 0.9, 0.1)  # 0.7, 0.8, 0.9   # 如果有discount则不能直接看cwpdis
        val_paras['batch_size'] = 2**np.arange(0,2)*256  # 64, 128, 256
        val_paras['tau'] = [0.01,0.02]
        val_paras['lr'] = [0.0005,0.001]
        
        val_res = eval("pd.DataFrame(itertools.product(" + ','.join(["val_paras['" + key + "']" for key in val_paras.keys()]) + "), columns = val_paras.keys())")
        
        replay_buffer = utils.StandardBuffer(state_dim,  BATCH_SIZE, len(train_data), device)
        replay_buffer.add(np.array(train_data[state_col]), np.array(train_data['actions']).reshape(-1, 1), np.array(train_data[next_state_col]), np.array(train_data['reward']).reshape(-1, 1), np.array(train_data['done']).reshape(-1, 1))
    
    
        # select best hyperparameters
        for ind in val_res.index:
            parameters.update(val_res.loc[ind,val_paras.keys()])
            loss, policy, Q_s, final_actions = train_BCQ(replay_buffer,  num_actions, state_dim, device, parameters)
            Q_s_, final_actions_ = policy.select_action_new(np.array(val_data[state_col]))
            val_dt = pd.concat([val_data[['patientunitstayid','step_id','actions','reward']] , pd.DataFrame(final_actions_, columns = ['ai_action'])], axis = 1)
            v_cwpdis, ess, fcs = cal_cpwdis(val_dt, parameters)
            val_res.loc[ind, 'v_cwpdis'] = v_cwpdis
            val_res.loc[ind, 'ess'] = ess
            val_res.loc[ind, 'fcs'] = fcs
            
        best_ind = np.argmax(val_res['v_cwpdis'])
        parameters.update(val_res.loc[best_ind,val_paras.keys()])
        
        val_str = 'val'
        
    else:
        val_res = pd.DataFrame()
        val_str = 'no-val'
        
    # Initialize buffer
    train_val_data = pd.concat([train_data, val_data])
    train_val_data = train_val_data.reset_index(drop = True)
    replay_buffer = utils.StandardBuffer(state_dim,  BATCH_SIZE, len(train_val_data), device)
    replay_buffer.add(np.array(train_val_data[state_col]), np.array(train_val_data['actions']).reshape(-1, 1), np.array(train_val_data[next_state_col]), np.array(train_val_data['reward']).reshape(-1, 1), np.array(train_val_data['done']).reshape(-1, 1))

    loss, policy, Q_s, final_actions = train_BCQ(replay_buffer, num_actions, state_dim, device, parameters)
    if torch.cuda.is_available():
        loss_ = [i.detach().cpu().numpy() for i in loss]
    else:   
        loss_ = [i.detach().numpy() for i in loss]
    
    data_ = pd.concat([train_val_data, pd.DataFrame(Q_s, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    data_['ai_action'] = final_actions
    
    # save model
    mod_dir = '../model/%s_%s_%s_%s_%s_%s_%s_trainon%s_crit-%s/'%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')[2:16], REWARD_FUN, str(SEED), MODEL, val_str, STEP_LENGTH, CUT_TIME, TRAIN_SET, str(CRITICAL_STATE))
    
    
    if os.path.isdir(mod_dir) == False:
        os.makedirs(mod_dir)            
    torch.save(policy, mod_dir + 'model.pkl')
    # load
    # policy = torch.load('\model.pkl')
    
    res_dir_ = '../result/%s_%s_%s_%s_%s_%s_%s_trainon%s_crit-%s/'%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')[2:16], REWARD_FUN, str(SEED), MODEL, val_str, STEP_LENGTH, CUT_TIME, TRAIN_SET, str(CRITICAL_STATE))
    
    #  evaluate on train(+val) set
    datatype = TRAIN_SET
    evaluation_new.run_eval(res_dir_, data_,loss_,datatype, SEED, 'train', parameters, val_str, val_res)
    
    # evaluate on test_in set
    Q_s_, final_actions_ = policy.select_action_new(np.array(test_data[state_col]))
    test_data = pd.concat([test_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    test_data['ai_action'] = final_actions_
    evaluation_new.run_eval(res_dir_, test_data,False,datatype, SEED, 'innertest', parameters, val_str, pd.DataFrame())
    
    # evaluate on test_out set
    outer_test_data = pd.read_csv(data_for_train_test['test'])
    
    outer_test_data = pre_processing(outer_test_data, TEST_SET)
    
    Q_s_, final_actions_ = policy.select_action_new(np.array(outer_test_data[state_col]))
    
    outer_test_data = pd.concat([outer_test_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    outer_test_data['ai_action'] = final_actions_
    
    datatype_ = TEST_SET
    evaluation_new.run_eval(res_dir_, outer_test_data,False,datatype_, SEED, 'outtertest', parameters, val_str, pd.DataFrame())
    
    # aa = data.loc[0:1000, state_col + next_state_col+['done']]
    # aaa = data['ori_sofa_24hours']
    # tt = data['reward']
    # cwpdis_ess_eval(outer_test_data)
    # data = outer_test_data
    #  xx = outer_test_data['reward']
    #  aa =outer_test_data[outer_test_data['reward'].astype(str) == 'nan']['step_id']
    # tmp = outer_test_data.loc[26031:26034, ['done','ori_spo2']]
    # xxx = pd.isnull(train_data['ori_spo2'])