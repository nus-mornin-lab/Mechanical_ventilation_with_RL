import argparse
import copy
import importlib
import json
import os
#  os.chdir('D:/pingan/比赛/2020SingaporeDatathon/BCQ/BCQ-master/discrete_BCQ')
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


state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
ITERATION_ROUND = setting.ITERATION_ROUND
ACTION_SPACE = setting.ACTION_SPACE
BATCH_SIZE = setting.BATCH_SIZE
SEED = setting.SEED
REWARD_FUN = setting.REWARD_FUN
TIME_RANGE = setting.TIME_RANGE

# MODEL = setting.MODEL

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
    
    eicu_data = 'data/data_rl_60min_eicu_'+ TIME_RANGE + '.csv'#'data/data_rl_with_dose.csv' #
    mimic_data = 'data/data_rl_60min_mimic_'+ TIME_RANGE +'.csv'#'data/mimic_data_rl_with_dose_11Dec.csv' 
    cut_mimic_data = 'data/data_rl_60min_mimic_'+ TIME_RANGE + '_72h' +'_cut.csv'
    cut_eicu_data = 'data/data_rl_60min_eicu_'+ TIME_RANGE + '_72h' +'_cut.csv'
    
    ### train val test on mimic
    if os.path.exists(cut_mimic_data) == False:
        data = pd.read_csv(mimic_data)
        data['gender'] = data['gender'].apply(lambda x: 0 if (x == 'F' or x == 0)  else 1) # 男-1， 女-0
        data['step_id'] = data['step_id'].astype(int)
        data = data[data['step_id'] < 72]  # 72小时以内的数据
        data = data.reset_index(drop = True)
        
        # cur train val test 
        all_stay = np.array(data['patientunitstayid'].unique().tolist()) 
        val_test_stay = np.random.choice(all_stay, size=int(len(all_stay)*2/5), replace=False)
        val_stay = np.random.choice(val_test_stay, size=int(len(val_test_stay)/2), replace=False).tolist()
        test_stay = list(set(val_test_stay) ^ set(val_stay))
        train_stay = list(set(all_stay) ^ set(val_test_stay))
        
        data['set'] = data['patientunitstayid'].apply(lambda x: 'trainset' if x in train_stay else 'testset' if x in test_stay else 'valset' if x in val_stay else np.nan)
        data.to_csv(cut_mimic_data)
        
    ### test on eicu
    if os.path.exists(cut_eicu_data) == False:
        data = pd.read_csv(eicu_data)
        data['gender'] = data['gender'].apply(lambda x: 0 if (x == 'F' or x == 0)  else 1) # 男-1， 女-0
        data['step_id'] = data['step_id'].astype(int)
        data = data[data['step_id'] < 72]  # 72小时以内的数据
        data = data.reset_index(drop = True)
        data.to_csv(cut_eicu_data)
    
    data = pd.read_csv(cut_mimic_data)
    # data['step_id'] = data['step_id'].astype(int)
    
    # calculate reward and actions
    data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
    data['actions'] = data.apply(lambda x: int(x[action_dis_col[0]] * 9 + x[action_dis_col[1]] * 3 + x[action_dis_col[2]]), axis =1)
    actions = data['actions']

    data.fillna(0, inplace=True)    
    MAX_TIMESTEPS = int(len(data) / BATCH_SIZE * ITERATION_ROUND)
    GAMMA = setting.GAMMA

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
    
    if setting.VALIDATION == True:
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
    mod_dir = 'model/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + setting.REWARD_FUN  +'_' + str(SEED) + '_' + setting.MODEL + '_' + val_str + '_'+ TIME_RANGE + '_72h_cut' +'/'
    if os.path.isdir(mod_dir) == False:
        os.makedirs(mod_dir)            
    torch.save(policy, mod_dir + 'model.pkl')
    # load
    # policy = torch.load('\model.pkl')
    
    
    res_dir_ = 'result/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + setting.REWARD_FUN  +'_' + str(SEED) + '_' + setting.MODEL + '_' + val_str + '_'+ TIME_RANGE + '_72h_cut' +'/'
    #  evaluate on train(+val) set
    datatype = 'mimic'
    evaluation_new.run_eval(res_dir_, data_,loss_,datatype, setting.SEED, 'train', parameters, val_str, val_res, TIME_RANGE)
    
    # evaluate on test_in set
    Q_s_, final_actions_ = policy.select_action_new(np.array(test_data[state_col]))
    test_data = pd.concat([test_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    test_data['ai_action'] = final_actions_
    evaluation_new.run_eval(res_dir_, test_data,False,datatype, setting.SEED, 'innertest', parameters, val_str, pd.DataFrame(), TIME_RANGE)
    
    # evaluate on test_out set
    outer_test_data = pd.read_csv(cut_eicu_data)
    
    outer_test_data['reward'] = outer_test_data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
    outer_test_data['actions'] = outer_test_data.apply(lambda x: int(x[action_dis_col[0]] * 9 + x[action_dis_col[1]] * 3 + x[action_dis_col[2]]), axis =1)
    outer_test_data.fillna(0, inplace=True)
    if len(outer_test_data['PEEP_level'].unique()) == 3:
        outer_test_data['PEEP_level'] = outer_test_data['PEEP_level'].apply(lambda x: 0 if (x == 0 or x == 1) else 1 if x == 2 else np.nan)

    Q_s_, final_actions_ = policy.select_action_new(np.array(outer_test_data[state_col]))
    outer_test_data = pd.concat([outer_test_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    outer_test_data['ai_action'] = final_actions_
    
    datatype_ = 'eicu'
    evaluation_new.run_eval(res_dir_, outer_test_data,False,datatype_, setting.SEED, 'outtertest', parameters, val_str, pd.DataFrame(), TIME_RANGE)
    
    # len(train_data)
    
    # len(val_data)
    
    # len(test_data)
    
    # len(outer_test_data)
    
    # len(outer_test_data['patientunitstayid'].unique())
    
    # len(test_data['patientunitstayid'].unique())
    
    # len(train_data['patientunitstayid'].unique())
    
    # len(val_data['patientunitstayid'].unique())
    
    # len(test_data['patientunitstayid'].unique())
    
    # (outer_test_data['hosp_mort'] == 1).mean()
    
    # (outer_test_data.drop_duplicates(subset = ['patientunitstayid'])['hosp_mort'] == 1).mean()
    
    # (data.drop_duplicates(subset = ['patientunitstayid'])['hosp_mort'] == 1).mean()
    
    
    
