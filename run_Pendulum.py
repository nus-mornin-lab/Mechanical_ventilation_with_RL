"""
Dueling DQN & Natural DQN comparison

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import logging
import argparse
import tensorflow as tf

from RL_brain import DuelingDQN
# from evaluation import Evaluation
from evaluation_new import *


state_col = ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp', 'lactate',
             'bicarbonate', 'wbc', 'pao2', 'paco2', 'pH', 'gcs', 'intaketotal', 'nettotal',
             'urineoutput', 'med_sedation', 'med_neuromuscular_blocker', 'age', 'gender',
             'apache_iv', 'admissionweight', 'sofatotal', 'equivalent_mg_4h']
next_state_col = ['next_heartrate', 'next_respiratoryrate', 'next_spo2', 'next_temperature',
                  'next_sbp', 'next_dbp', 'next_lactate', 'next_bicarbonate', 'next_wbc',
                  'next_pao2', 'next_paco2', 'next_pH', 'next_gcs', 'next_intaketotal',
                  'next_nettotal', 'next_urineoutput', 'next_med_sedation', 'next_med_neuromuscular_blocker',
                  'next_age', 'next_gender', 'next_apache_iv', 'next_admissionweight', 'next_sofatotal', 'next_equivalent_mg_4h']
action_dis_col = ['PEEP_level', 'FiO2_level', 'Tidal_level']
reward_col = ['ori_spo2', 'next_ori_spo2', 'hosp_mort']


MEMORY_SIZE = 249579
ACTION_SPACE = 18# 27
STATE_DIM = len(state_col) # 24
BATCH_SIZE = 256
EPISODE = int(MEMORY_SIZE / BATCH_SIZE * 5)  # 遍历数据5轮。42w/256*5 = 8203
np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)

def train(RL, data, first_run=True):
    if first_run:
        # reward function
        data['reward'] = data.apply(lambda x: -1 if (x['done'] == 1 and x['hosp_mort'] == 1) else
                                            1  if (x['done'] == 1 and x['hosp_mort'] == 0) else
                                            0  if x['done'] == 0 else
                                            np.nan ,axis = 1)
        
        actions = data.apply(lambda x: action2onehot(x[action_dis_col]), axis =1)
        
        memory_array = np.concatenate([np.array(data[state_col]), 
                                    np.array(actions).reshape(-1, 1), 
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
    for i in tqdm(range(EPISODE)):
        RL.learn(i)


def action2onehot(action):
    # just like a Ternary conversion
    peep, fio2, tidal = 1 if action[0] == 2 else 0, action[1], action[2]
    index = 9 * int(peep) + 3 * int(fio2) + int(tidal)
    
    # action_vector = np.zeros(27, dtype=int)
    # action_vector[index] = 1

    return index

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, required=True)
    parser.add_argument('--first_run', '-f', type=bool, required=True)
    args = parser.parse_args()
    
    # read data
    data = pd.read_csv('data/data_rl_with_dose.csv')
    # data = pd.read_csv('../../data_rl.csv')
    data.fillna(0, inplace=True)
    print('\nLOAD DATA DONE!\n')
    print('data.shape', data.shape)
    # data = data.iloc[:MEMORY_SIZE]
    
    # split train and test set
    # length = data.shape[0]
    # train_data, test_data = data[:int(length*0.8)], data[int(length*0.8):]

    if args.mode == 'train':
        # init model
        sess = tf.Session()
        with tf.variable_scope('dueling'):
            dueling_DQN = DuelingDQN(
                n_actions=ACTION_SPACE, n_features=STATE_DIM, memory_size=MEMORY_SIZE,
                batch_size=BATCH_SIZE, e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

        sess.run(tf.global_variables_initializer())
        print_num_of_total_parameters(True, False)
        
        train(dueling_DQN, data, args.first_run)
        
        # save model
        saver = tf.train.Saver()
        saver.save(sess, 'models/duel_DQN')
        
    elif args.mode == 'eval':
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph('../model/model_LR_test.meta')
        new_saver.restore(sess,'../model/model_LR_test')#加载模型中各种变量的值，注意这里不用文件的后缀 

    # evaluate model
    eval_q = sess.run(dueling_DQN.q_eval, feed_dict={dueling_DQN.s: data[state_col]})
    print(np.where(eval_q<0))
    
    result_array = np.concatenate([data.values, eval_q], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(data.columns)+['Q_0', 'Q_1', 'Q_2', 'Q_3', 'Q_4', 'Q_5', 'Q_6', 'Q_7', 'Q_8', 'Q_9', 'Q_10', 'Q_11', 'Q_12', 'Q_13', 'Q_14', 'Q_15', 'Q_16', 'Q_17'])
                          # columns=list(data.columns)+['Q_0', 'Q_1', 'Q_2', 'Q_3', 'Q_4', 'Q_5', 'Q_6', 'Q_7', 'Q_8', 'Q_9', 'Q_10', 'Q_11', 'Q_12', 'Q_13', 'Q_14', 'Q_15', 'Q_16', 'Q_17', 'Q_18', 'Q_19', 'Q_20', 'Q_21', 'Q_22', 'Q_23', 'Q_24', 'Q_25', 'Q_26'])
    # result.to_csv('result.csv')
    
    print(eval_q.shape, type(eval_q))
    print(eval_q)
    
    
    # eval = Evaluation()
    run_eval(result)