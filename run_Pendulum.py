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
import setting

from RL_brain import DuelingDQN
# from evaluation import Evaluation
from evaluation_new import *

# setting
state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
ITERATION_ROUND = setting.ITERATION_ROUND
ACTION_SPACE = setting.ACTION_SPACE
BATCH_SIZE = setting.BATCH_SIZE
SEED = setting.SEED
REWARD_FUN = setting.REWARD_FUN

# drive
STATE_DIM = len(state_col) # 23 
np.random.seed(SEED)
tf.set_random_seed(SEED)
random.seed(SEED)

def train(RL, data, first_run=True):
    if first_run:
        # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
        
        actions = data.apply(lambda x: x[action_dis_col[0]] * 9 + x[action_dis_col[1]] * 3 + x[action_dis_col[2]], axis =1)
        
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
    EPISODE = int(MEMORY_SIZE / BATCH_SIZE * ITERATION_ROUND)
    for i in tqdm(range(EPISODE)):
        RL.learn(i)
    loss = RL.cost_his
    return loss


# def action2onehot(action):
#     # just like a Ternary conversion
#     peep, fio2, tidal = action[0], action[1], action[2]
#     index = 9 * int(peep) + 3 * int(fio2) + int(tidal)
    
#     action_vector = np.zeros(27, dtype=int)
#     action_vector[index] = 1

#     return index

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
    if len(data['PEEP_level'].unique()) == 3:
        data['PEEP_level'] = data['PEEP_level'].apply(lambda x: 0 if (x == 0 or x == 1) else 1 if x == 2 else np.nan)

    # data = pd.read_csv('../../data_rl.csv')
    data.fillna(0, inplace=True)
    print('\nLOAD DATA DONE!\n')
    print('data.shape', data.shape)
    
    MEMORY_SIZE = len(data)
    
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
        
        loss = train(dueling_DQN, data, args.first_run)
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
    run_eval(result, loss)