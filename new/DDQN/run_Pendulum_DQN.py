import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import logging
import argparse
import tensorflow as tf
import setting
import os
from datetime import datetime
import itertools
from RL_brain import DuelingDQN,FQI
import evaluation_new
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

# setting
state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
other_related_col = setting.other_related_col
other_related_next_col = setting.other_related_next_col
REWARD_FUN = setting.REWARD_FUN

def train_RL(sess, data_, num_actions, state_dim, MODEL, parameters):
    if MODEL == 'DQN':
        with tf.variable_scope('dueling'):
            RL_model = DuelingDQN(
                n_actions=num_actions,
                n_features=state_dim, 
                memory_size=len(data_),
                batch_size=parameters["batch_size"], 
                learning_rate = parameters["lr"],
                gamma = parameters["discount"],
                replace_target_iter = parameters["target_update_freq"],
                sess=sess, 
                dueling=True, 
                output_graph=True)
    # elif MODEL == 'FQI':
    #     with tf.variable_scope('dueling'):
    #         RL_model = FQI(
    #             n_actions=num_actions, n_features=state_dim, memory_size=len(train_val_data),
    #             batch_size=parameters["batch_size"], sess=sess, dueling=False, output_graph=True)
    
    sess.run(tf.global_variables_initializer())
        
    memory_array = np.concatenate([np.array(data_[state_col]), 
                            np.array(data_['actions']).reshape(-1, 1), 
                            np.array(data_['reward']).reshape(-1, 1), 
                            np.array(data_['done']).reshape(-1, 1),
                            np.array(data_[next_state_col])] ,
                            axis = 1)
    
    RL_model.store_transition(memory_array)
    
    # Initialize and load policy
    for i in tqdm(range(parameters["MAX_TIMESTEPS"])):
        RL_model.learn(i)
    loss = RL_model.cost_his

    Q_s = sess.run(RL_model.q_eval, feed_dict={RL_model.s: data_[state_col]})
    actions = np.argmax(Q_s, axis = 1)
    
    return loss ,RL_model, Q_s, actions

'''
def train(RL, data, first_run=True):
    if first_run:
        # reward function
        
        memory_array = np.concatenate([np.array(data[state_col]), 
                                    np.array(data['actions']).reshape(-1, 1), 
                                    np.array(data['reward']).reshape(-1, 1), 
                                    np.array(data['done']).reshape(-1, 1),
                                    np.array(data[next_state_col])] ,
                                    axis = 1)
        np.save('memory.npy', memory_array)
        
    else:
        memory_array = np.load(memory_array)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART TRAINING\n')
    if MODEL == 'DQN':
        EPISODE = int(MEMORY_SIZE / BATCH_SIZE * ITERATION_ROUND)
    elif MODEL == 'FQI':
        EPISODE = ITERATION_ROUND
        
    for i in tqdm(range(EPISODE)):
        RL.learn(i)
    loss = RL.cost_his
    return loss
'''


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

def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if output_detail:
            logging.info(parameters_string)
        logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print(parameters_string)
        print("\nTotal %d variables, %s params\n" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))

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
    mimic_set = pickle.load(open(mimic_set_path,'rb'))
    eicu_set = pickle.load(open(eicu_set_path,'rb'))
    patient_set['mimic'] = mimic_set
    patient_set['eicu'] = eicu_set
    
    data = pd.read_csv(data_for_train_test['train'])
    data = pre_processing(data, TRAIN_SET)
    
    MAX_TIMESTEPS = int(len(data) / BATCH_SIZE * ITERATION_ROUND)
    
    regular_parameters = {
    # Learning
    "MAX_TIMESTEPS":MAX_TIMESTEPS,
    "discount": GAMMA,  # 和其他model保持一致
    # "buffer_size": MEMORY_SIZE,   
    "batch_size": BATCH_SIZE,
    "lr": 1e-3,
    "polyak_target_update": True,  # 软/硬更新
    "target_update_freq": 50, 
    "tau": 0.01
    }
    
    # Make env and determine properties
    state_dim = len(state_col)
    num_actions = ACTION_SPACE
    parameters = regular_parameters
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
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
        val_paras["discount"] = [0.99,0.995]
        val_paras['lr'] = [0.0005,0.001]
        
        val_res = eval("pd.DataFrame(itertools.product(" + ','.join(["val_paras['" + key + "']" for key in val_paras.keys()]) + "), columns = val_paras.keys())")
        
        for ind in val_res.index:
            parameters.update(val_res.loc[ind,val_paras.keys()])
            sess = tf.Session()
            loss, policy, Q_s, final_actions = train_RL(sess, train_data, num_actions, state_dim, MODEL, parameters)
            Q_s_ = sess.run(policy.q_eval, feed_dict={policy.s: val_data[state_col]})
            final_actions_ = np.argmax(Q_s_, axis = 1)
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
    
    sess = tf.Session()
    
    loss, policy, Q_s, final_actions = train_RL(sess, train_val_data, num_actions, state_dim, MODEL, parameters)

    data_ = pd.concat([train_val_data, pd.DataFrame(Q_s, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    data_['ai_action'] = final_actions
    
    # save model
    mod_dir = '../model/%s_%s_%s_%s_%s_%s_%s_trainon%s_crit-%s/'%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')[2:16], REWARD_FUN, str(SEED), MODEL, val_str, STEP_LENGTH, CUT_TIME, TRAIN_SET, str(CRITICAL_STATE))
    if os.path.isdir(mod_dir) == False:
        os.makedirs(mod_dir)            
    saver = tf.train.Saver()
    saver.save(sess, mod_dir+'model')
    # load
    # sess = tf.Session()
    # new_saver = tf.train.import_meta_graph('models/duel_DQN.meta')
    # new_saver.restore(sess, 'models/duel_DQN')#加载模型中各种变量的值，注意这里不用文件的后缀 

    res_dir_ = '../result/%s_%s_%s_%s_%s_%s_%s_trainon%s_crit-%s/'%(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')[2:16], REWARD_FUN, str(SEED), MODEL, val_str, STEP_LENGTH, CUT_TIME, TRAIN_SET, str(CRITICAL_STATE))

    #  evaluate on train(+val) set
    datatype = TRAIN_SET
    evaluation_new.run_eval(res_dir_, data_,loss, datatype, SEED, 'train', parameters, val_str, val_res)

    # evaluate on test_in set
    Q_s_ = sess.run(policy.q_eval, feed_dict={policy.s: test_data[state_col]})
    final_actions_ = np.argmax(Q_s_, axis = 1)
    test_data = pd.concat([test_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    test_data['ai_action'] = final_actions_
    evaluation_new.run_eval(res_dir_, test_data,False,datatype, SEED, 'innertest', parameters, val_str, pd.DataFrame())

    # evaluate on test_out set
    outer_test_data = pd.read_csv(data_for_train_test['test'])
    
    outer_test_data = pre_processing(outer_test_data, TEST_SET)
    
    Q_s_ = sess.run(policy.q_eval, feed_dict={policy.s: outer_test_data[state_col]})
    final_actions_ = np.argmax(Q_s_, axis = 1)

    outer_test_data = pd.concat([outer_test_data, pd.DataFrame(Q_s_, columns = ['Q_' + str(i) for i in range(num_actions)])] , axis = 1)
    outer_test_data['ai_action'] = final_actions_
    
    datatype_ = TEST_SET
    evaluation_new.run_eval(res_dir_, outer_test_data,False,datatype_, SEED, 'outtertest', parameters, val_str, pd.DataFrame())

    # pd.Series(np.argmax(Q_s,axis = 1)).value_counts()
    '''
    def cal_ress(dt):
        dt['mean_reward'] = dt['reward'].mean()
        return dt['mean_reward'].values[0]
    '''
    # train_val_data.groupby('actions').apply(cal_ress)
