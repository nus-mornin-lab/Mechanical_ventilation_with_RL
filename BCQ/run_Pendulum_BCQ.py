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
import evaluation_fin
from sklearn.model_selection import KFold
import random
import itertools
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import torch.utils.data as Data

SEED = setting.SEED
ITERATION_ROUND = setting.ITERATION_ROUND
ACTION_SPACE = setting.ACTION_SPACE
BATCH_SIZE = setting.BATCH_SIZE
GAMMA = setting.GAMMA
VALIDATION = setting.VALIDATION
ROUND_PER_EVAL = setting.ROUND_PER_EVAL

DATA_DATE = setting.DATA_DATE
TIME_RANGE = setting.TIME_RANGE
CUT_TIME = setting.CUT_TIME 
TRAIN_SET = setting.TRAIN_SET
STEP_LENGTH = setting.STEP_LENGTH
CRITICAL_STATE = setting.CRITICAL_STATE
TOP_STATE = setting.TOP_STATE
MISSING_CUT = setting.MISSING_CUT
PRETRAIN = setting.PRETRAIN 
RE_DIVIDE_SET = setting.RE_DIVIDE_SET
VAR = setting.VAR


MODEL = setting.MODEL

setting_paras = setting.setting_paras
setting_iter = eval("pd.DataFrame(itertools.product(" + ','.join(["setting_paras['" + key + "']" for key in setting_paras.keys()]) + "), columns = setting_paras.keys())")


state_col = setting.state_col
next_state_col = setting.next_state_col
ori_state_col = setting.ori_state_col
action_dis_col = setting.action_dis_col
other_related_col = setting.other_related_col
other_related_next_col = setting.other_related_next_col
REWARD_FUN = setting.REWARD_FUN


# rnd = 0
def evaluate_on_3_set(policy, mod_dir, res_dir_, rnd, train_val_data, test_data, outer_test_data, num_actions, state_col,TRAIN_SET,TEST_SET,parameters,val_str, val_res):
    # save model
    # torch.save(policy, mod_dir + 'model%s.pkl'%(str(rnd)))
    #  evaluate on train(+val) set
    Q_s_, final_actions_ = policy.select_action_new(np.array(train_val_data[state_col]))
    data_ = pd.concat([train_val_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    data_['ai_action'] = final_actions_
    datatype = TRAIN_SET
    # evaluation_new.run_eval(res_dir_, data_,False,datatype, SEED, 'train%s'%(str(rnd)), parameters, val_str, val_res)
    
    # evaluate on test_in set
    Q_s_, final_actions_ = policy.select_action_new(np.array(test_data[state_col]))
    data_ = pd.concat([test_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    data_['ai_action'] = final_actions_
    # evaluation_new.run_eval(res_dir_, data_,False,datatype, SEED, 'innertest%s'%(str(rnd)), parameters, val_str, pd.DataFrame())
    
    # evaluate on test_out set
    Q_s_, final_actions_ = policy.select_action_new(np.array(outer_test_data[state_col]))
    data_2 = pd.concat([outer_test_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    data_2['ai_action'] = final_actions_
    datatype_ = TEST_SET
    # evaluation_new.run_eval(res_dir_, data_2,False,datatype_, SEED, 'outtertest%s'%(str(rnd)), parameters, val_str, pd.DataFrame())
    
    # evaluate the groupby_result
    res_quan = evaluation_fin.run_eval(res_dir_, data_, data_2,False,datatype_, SEED, 'test%s'%(str(rnd)), parameters, val_str, pd.DataFrame())
    return res_quan

def plot_loss(res_dir_, policy):
    # plot loss
    plt.figure(figsize=(7,4))
    plt.plot(policy.loss)
    plt.savefig(res_dir_ + 'loss.jpg',dpi = 100)
    
    plt.figure(figsize=(7,4))
    plt.plot(policy.pretrain_loss)
    plt.savefig(res_dir_ + 'pretrain_loss.jpg',dpi = 100)
    
    plt.figure(figsize=(7,4))
    plt.plot(policy.q_loss)
    plt.savefig(res_dir_ + 'q_loss.jpg',dpi = 100)
    
    plt.figure(figsize=(7,4))
    plt.plot(policy.i_loss)
    plt.savefig(res_dir_ + 'i_loss.jpg',dpi = 100)
    
    plt.figure(figsize=(7,4))
    plt.plot(policy.i2_loss)
    plt.savefig(res_dir_ + 'i2_loss.jpg',dpi = 100)

def init_model(replay_buffer, num_actions, state_dim, device, parameters ):
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
    try:
        pret = parameters['PRETRAIN']
    except:
        pret = PRETRAIN
        
    if pret == True:
        # pretrain
        print ('pretraining...' )
        for turn in range(int(parameters["MAX_TIMESTEPS"]/ITERATION_ROUND*2)):
            policy.pretrain(replay_buffer)
        # copy target
        policy.Q_target.load_state_dict(policy.Q.state_dict())
        print ('pretraining done' )
    return policy

# Trains BCQ offline
def train_BCQ_1_round(policy, replay_buffer, parameters):
    training_iters = 0
    
    while training_iters < parameters["STEP_PER_ROUND"]: 
        # print ('round: ' + str(training_iters))
        policy.train(replay_buffer)
        training_iters += 1
    return policy

def divide_set(data_name, save_path):
    # np.random.seed(523)
    # random.seed(523)
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

# len(data['patientunitstayid'].unique())
def pre_processing(data, SET, MODE, norm_dict = {}, VAR=''):
    # SET = TRAIN_SET

    # tag some labels...
    data['set'] = data['patientunitstayid'].apply(lambda x: patient_set[SET][x])
    data['step_id'] = data['step_id'].astype(int)
    data['done'] = 0
    
    # data['step_id'] = data['step_id'].astype(int)
    # if 'phys_action' not in data.columns.tolist():
    data['phys_action'] = data.apply(setting.tag_action,action_dis_col = action_dis_col,axis = 1)
    # first cut length (if need)
    if VAR == '':
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
        elif CUT_TIME == '24h':
            if STEP_LENGTH == '240min':
                data = data[data['step_id'] < 6]
            elif STEP_LENGTH == '60min':
                data = data[data['step_id'] < 24]
        elif CUT_TIME == '7d':
            if STEP_LENGTH == '240min':
                data = data[data['step_id'] < 42]
            elif STEP_LENGTH == '60min':
                data = data[data['step_id'] < 168]
    else:
        if CUT_TIME == '72h':
            data = data[data['time'] < 72 * 60]
        elif CUT_TIME == '48h':
            data = data[data['time'] < 48 * 60]
        elif CUT_TIME == '24h':
            data = data[data['time'] < 24 * 60]
        elif CUT_TIME == '7d':
            data = data[data['time'] < 7*24 * 60]
    
    # then cut missing1
    if MISSING_CUT == True:
        data = data[data['PEEP_missing1'] == 0]
        data = data[data['FiO2_missing1'] == 0]
        data = data[data['Tidal_missing1'] == 0]
    
    # then do normalization
    if MODE == 'train':
        norm_dict = {}
        for ori_col in ori_state_col:
            col = ori_col.replace('ori_','')
            assert col in state_col
            mean_ = np.nanmean(data[ori_col])
            std_ = np.nanstd(data[ori_col])
            norm_dict[col] = [mean_, std_]
            data[col] = data[ori_col].apply(lambda x: np.nan if pd.isnull(x) else (x - mean_)/std_).tolist()
        
    else:
        for ori_col in ori_state_col:
            col = ori_col.replace('ori_','')
            assert col in state_col
            mean_ = norm_dict[col][0]
            std_ = norm_dict[col][1]
            data[col] = data[ori_col].apply(lambda x: np.nan if pd.isnull(x) else (x - mean_)/std_).tolist()

        
    # then fill data and shift to tag next_col
    def f_b_fill_and_tag_done(dt):
        dt['done'].values[-1] = 1
        dt[state_col+other_related_col] = dt[state_col+other_related_col].fillna(method = 'ffill').fillna(method = 'bfill')
        return dt
        # first f_b_fill 
    data = data.groupby(['patientunitstayid']).apply(f_b_fill_and_tag_done)
    
        # then fill median
    data[state_col+other_related_col] = data[state_col+other_related_col].apply(lambda x: x.fillna(x.median()))
    
        # then shift to tag next_col
    for next_ori_col in other_related_next_col:
        if next_ori_col not in data.columns.tolist():
            data[next_ori_col] = np.nan
    data[next_state_col+other_related_next_col] = data[state_col+other_related_col].shift(-1)
            
    # then modify next_col
    data.loc[data['done'] == 1, other_related_next_col] = np.nan
    data.loc[data['done'] == 1, next_state_col] = 0
    # medians = data[['patientunitstayid'] + state_col+other_related_col+next_state_col+other_related_next_col].groupby('patientunitstayid').agg(np.nanmedian).agg(np.nanmedian)
    # data[state_col+other_related_col+next_state_col+other_related_next_col] = data[state_col+other_related_col+next_state_col+other_related_next_col].apply(lambda x: x.fillna(medians[x.name])) 
    # data[state_col+other_related_col+next_state_col+other_related_next_col] = data[state_col+other_related_col+next_state_col+other_related_next_col].apply(lambda x: x.fillna(medians[x.name])) 
    # data.loc[data['done'] == 1, next_state_col+other_related_next_col] = np.nan
    
    data = data.reset_index(drop = True)
    
    # tag reach
    data['spo2_reach'] = data['next_ori_spo2'].apply(lambda x: np.nan if pd.isnull(x) else 1 if (x >= 94 and x <= 98) else 0 )
    data['mbp_reach'] = data['next_ori_mbp'].apply(lambda x: np.nan if pd.isnull(x) else 1 if (x >= 70 and x <= 80) else 0 )

    # finally calculate reward and actions
    # data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
    data['actions'] = data['phys_action']
    # data.fillna(0, inplace=True)    
    return data, norm_dict

# calculate cwpdis
def cal_metrics(val_dt, parameters):
    # val_dt = test_data
    val_dt['concordant'] = val_dt.apply(lambda x: (x[action_dis_col[0]] == x['ai_action']) +0 ,axis = 1)
    val_dt['conc_cumsum'] = val_dt.groupby(['patientunitstayid'])['concordant'].cumsum()
    # print (val_dt['conc_cumsum'])
    val_dt['pnt'] = (val_dt['conc_cumsum'] == (val_dt['step_id'] + 1))+0
    
    v_cwpdis = 0
    fcs = 0
    
    val_dt['valid_reward'] = (val_dt['reward']*val_dt['pnt'])
    # val_dt.groupby(['step_id']).apply(lambda dt: print (sum(dt['valid_reward'])))
    bb = val_dt.groupby(['step_id']).apply(lambda dt: sum(dt['valid_reward'])/ sum(dt['pnt']) if sum(dt['pnt'])>0 else 0.0)
    bb = pd.DataFrame(bb)
    bb['step_id'] = bb.index
    # print (bb.shape)
    # print (bb)
    cc = bb.apply(lambda x: parameters['discount']**(x['step_id']) * x[0] , axis = 1)
    
    v_cwpdis = sum(cc)
    ess = sum(val_dt['pnt'])
    fcs = sum(val_dt[val_dt['step_id'] == max(val_dt['step_id'])]['pnt'])
    
    conc_rate = np.nanmean(val_dt['concordant'])
    
    return v_cwpdis, ess, fcs, conc_rate

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
    first = 1
    last_data_setting = '.'
    considered_list = ['TRAIN_SET','CUT_TIME','STEP_LENGTH','MISSING_CUT','VAR','SEED'] if RE_DIVIDE_SET else ['TRAIN_SET','CUT_TIME','STEP_LENGTH','MISSING_CUT','VAR']
    considered_data_setting = list(set(setting_paras.keys()).intersection(set(considered_list)))
    
    # save path
    mod_dir = '../model/%s/'%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')[2:16])
    res_dir_ = '../result/%s/'%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')[2:16])
    if os.path.isdir(mod_dir) == False:
        os.makedirs(mod_dir)            
    if os.path.isdir(res_dir_) == False:
        os.makedirs(res_dir_)   
    
    # ind = 3
    for ind in setting_iter.index:
        
        cur_setting = setting_iter.loc[ind,setting_paras.keys()]
        cur_data_setting = ' '.join([str(cur_setting[k]) for k in considered_data_setting])
        for k in setting_paras.keys():
            exec(k + " = cur_setting['" + k +"']")

        if last_data_setting != cur_data_setting:
            last_data_setting = cur_data_setting

            mimic_data = '../data/%s/data_rl_%s_mimic%s_norm.csv'%(DATA_DATE, STEP_LENGTH, VAR) #'data/mimic_data_rl_with_dose_11Dec.csv' 
            eicu_data = '../data/%s/data_rl_%s_eicu%s_norm.csv'%(DATA_DATE, STEP_LENGTH, VAR) #'data/data_rl_with_dose.csv' #
            if RE_DIVIDE_SET == True:
                mimic_set_path = '../data/%s/mimic_set_%s.pkl'%(DATA_DATE, str(SEED))
                eicu_set_path = '../data/%s/eicu_set_%s.pkl'%(DATA_DATE, str(SEED))
            else:
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
            
            # torch.manual_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            
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
        
            # train val inner test data
            data = pd.read_csv(data_for_train_test['train'])
            data,norm_dict = pre_processing(data, TRAIN_SET, 'train',{},VAR)
            
            # data.to_csv('eicu_preprocessed.csv')
            # norm_dict
            
            # outer test data
            outer_test_data = pd.read_csv(data_for_train_test['test'])
            outer_test_data, _ = pre_processing(outer_test_data, TEST_SET, 'test', norm_dict,VAR)
            # outer_test_data.to_csv('mimic_preprocessed.csv')
            
            with open(res_dir_+ 'norm_dict_%s.pkl'%(cur_data_setting),'wb') as fw:
                pickle.dump(norm_dict,fw)
        
        #  aa = data[state_col + action_dis_col]
        MAX_TIMESTEPS = int(len(data)*0.2 / BATCH_SIZE * ITERATION_ROUND)
        # MAX_TIMESTEPS = int(len(data)*0.8 / BATCH_SIZE * ITERATION_ROUND)
        STEP_PER_ROUND = int(len(data)*0.8 / BATCH_SIZE)
        
        regular_parameters = {
        "is_atari":False,
        # Learning
        "MAX_TIMESTEPS":MAX_TIMESTEPS,
        "STEP_PER_ROUND":STEP_PER_ROUND,
        "discount": GAMMA,  # 和其他model保持一致
        "BCQ_THRESHOLD": 0.1,
        # "buffer_size": MEMORY_SIZE,   
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "lr": 3e-4,
        "polyak_target_update": True,  # 软/硬更新
        "target_update_freq": 50, 
        "tau": 0.01
        }
        
        regular_parameters.update(cur_setting)
        
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
    
        data['reward'] = data.apply(eval('setting.' +REWARD_FUN),a = regular_parameters['a'], b= regular_parameters['b'], c= regular_parameters['c'], e= regular_parameters['e'] , f= regular_parameters['f'], axis = 1)
        outer_test_data['reward'] = outer_test_data.apply(eval('setting.' +REWARD_FUN),a = regular_parameters['a'], b= regular_parameters['b'], c= regular_parameters['c'], e= regular_parameters['e'] , f= regular_parameters['f'], axis = 1)
        
                               
        # Make env and determine properties
        state_dim = len(state_col)
        num_actions = ACTION_SPACE
        parameters = regular_parameters
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
        
        
            # select best hyperparameters - not used now
            for ind in val_res.index:
                parameters.update(val_res.loc[ind,val_paras.keys()])
                loss, policy, Q_s, final_actions, q_loss, i_loss, i2_loss = train_BCQ_1_round(replay_buffer,  num_actions, state_dim, device, parameters)
                Q_s_, final_actions_ = policy.select_action_new(np.array(val_data[state_col]))
                val_dt = pd.concat([val_data[['patientunitstayid','step_id','actions','reward']] , pd.DataFrame(final_actions_, columns = ['ai_action'])], axis = 1)
                # v_cwpdis, ess, fcs = cal_cpwdis(val_dt, parameters)
                v_cwpdis, ess, fcs, conc_rate = cal_metrics(val_dt, parameters)
                
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
        
    
        
        policy = init_model(replay_buffer, num_actions, state_dim, device, parameters )
        
        res_quan = evaluate_on_3_set(policy, mod_dir, res_dir_, 0, train_val_data, test_data, outer_test_data, num_actions, state_col,TRAIN_SET,TEST_SET,parameters,val_str, val_res)
        rd = 0
        
        torch.save(policy, mod_dir + 'model_%s.pkl'%((' ').join(str(v) for v in cur_setting.tolist())))
        
        for rnd in range(1,ITERATION_ROUND+1) :
            print ('iter_',rnd)
            policy = train_BCQ_1_round(policy, replay_buffer, parameters)
            if (rnd) % ROUND_PER_EVAL == 0:
                res_quan_ = evaluate_on_3_set(policy, mod_dir, res_dir_, rnd, train_val_data, test_data, outer_test_data, num_actions, state_col,TRAIN_SET,TEST_SET,parameters,val_str, val_res)
                res_quan = max(res_quan, res_quan_)
                if res_quan_ == res_quan:
                    rd = rnd
                    torch.save(policy, mod_dir + 'model_%s.pkl'%((' ').join(str(v) for v in cur_setting.tolist())))
        
        if first == 1:
            f = open(res_dir_ + 'stats.txt', 'a')
            f.write('best score, round,  -- %s\n'%((' ').join(setting_iter.columns.tolist())))
            f.close()             
            first = 0
        
        f = open(res_dir_ + 'stats.txt', 'a')
        f.write('best score:%s round:%s -- %s\n'%(str(res_quan),str(rd),(' ').join(str(v) for v in cur_setting.tolist())))
        f.close()


