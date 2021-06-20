# -*- coding: utf-8 -*-
"""
@author: Zhuoyang XU
"""


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats 
import os
import time
from datetime import datetime
import setting
import json

def run_eval(res_dir_, data,data2, loss, datatype, SEED = setting.SEED, mode = 'train', parameters = {},val_str = 'val', val_res = pd.DataFrame(), TIME_RANGE = '24h'):

    def tag_conc_rate_and_diff_mean(dt):
        for v in action_types:
            dt[v + '_conc_rate'] = dt[v + '_conc'].mean()
            dt[v + '_diff_mean'] = dt[v + '_diff'].mean()
        return dt
    
    def discre_conc_level(x):
        xx = [0.1,0.3,0.5,0.7,0.9]
        for t in xx:
            if x>= t-0.1 and x < t+0.1:
                return t
        if x == 1:
            return 0.9
        
    def discre_diff_level(x):
        xx = [-2,-1,0,1,2]
        for t in xx:
            if x>= t-0.5 and x < t+0.5:
                return t
        if x == 2:
            return 2
    

    def diff_vs_outcome2(data,data2, outcome_col1 = 'hosp_mort',outcome_col2 = 'spo2_reach',outcome_col3 = 'mbp_reach', bootstrap_round = 1):
    
    # vs_motality
        def derive_x_s_res(data):
            data = data.reset_index(drop = True).copy()
            # 4-hour level 
            x_s = {}
            res = {}
            res_ = {}
            res__ = {}
            for v in action_types:
                x_s[v] = sorted(set(data[v + '_diff']))
                res[v] = {}
                res_[v] = {}
                res__[v] = {}
                for k in x_s[v]:
                    res[v][k] = []
                    res_[v][k] = []
                    res__[v][k] = []
                
            used_data = data[[v + '_diff' for v in action_types] + [outcome_col1, outcome_col2, outcome_col3 ]]
            for i in range(bootstrap_round):
                # if i%10 == 0:
                    # print ('bootstrap 4-hour level: ' + str(i) + '...')
                if bootstrap_round == 1:
                    df_index = used_data.index.tolist()
                else:
                    df_index = np.random.choice(used_data.index, size = len(used_data))
                # visit level
                # diff vs motality
                for v in action_types:
                    # v = 'PEEP'
                    diff_col = v + '_diff'
                    # aa = used_data.loc[df_index].groupby(diff_col).apply(lambda dt: dt[outcome_col1].mean())
                    aa1 = used_data.loc[df_index].groupby(diff_col).apply(lambda dt: dt[[outcome_col1,outcome_col2,outcome_col3]].mean())
                    for k in x_s[v]:
                        res[v][k].append(aa1.loc[k,outcome_col1])
                        res_[v][k].append(aa1.loc[k,outcome_col2])
                        res__[v][k].append(aa1.loc[k,outcome_col3])
            # print ('bootstrap 4-hour level done ...')

            return x_s, res, res_, res__
        
        x_s, res, res_, res__ = derive_x_s_res(data)
        x_s2, res2, res_2, res__2 = derive_x_s_res(data2)
        
        def judge_right(x_s,res):
            right_cnt =0
            for v in action_types:
                x_s_v = x_s[v]
                min_v = 100
                for k in x_s_v:
                    if k == 0:
                        mid_v = res[v][k][0]
                    else:
                        min_v = min(min_v, res[v][k][0])
                    
                if mid_v < min_v:
                    right_cnt +=1
            return right_cnt
        
        def judge_right_2(x_s,res):
            right_cnt =0
            for v in action_types:
                x_s_v = x_s[v]
                max_v = -1
                for k in x_s_v:
                    if k == 0:
                        mid_v = res[v][k][0]
                    else:
                        max_v = max(max_v, res[v][k][0])
                    
                if mid_v > max_v:
                    right_cnt +=1
            return right_cnt
        
        res_quan = 100*judge_right(x_s,res) + 100*judge_right(x_s2,res2) + judge_right_2(x_s,res_) + judge_right_2(x_s2,res_2) + judge_right_2(x_s,res__) + judge_right_2(x_s2,res__2)
        
        return res_quan
    
    def tag_data(data):
        data['ai_Q'] = np.max(data[Q_list],axis = 1)
        if ai_action not in data.columns.tolist():
            data[ai_action] = np.argmax(np.array(data[Q_list]),axis = 1)
        data[phys_Q] = data.apply(lambda x: x['Q_' + str(int(x[phys_action]))], axis = 1)
        
        data[action_types[0] + '_level_ai'] = (data[ai_action]/9).apply(lambda x: int(x))
        data[action_types[1] + '_level_ai'] = (data[ai_action]%9/3).apply(lambda x: int(x))
        data[action_types[2] + '_level_ai'] = (data[ai_action]%9%3).apply(lambda x: int(x))
        
        
        for v in action_types:
            data[v + '_diff'] = data[v + '_level'] - data[v + '_level_ai']
            data[v + '_conc'] = (data[v + '_level'] == data[v + '_level_ai']) + 0
            
        data = data.reset_index(drop = True).copy()
        
        return data
    
    action_num = 18
    Q_list = ['Q_' + str(i) for i in range(action_num)]
    action_types = ['PEEP', 'FiO2', 'Tidal']
    phys_Q = 'phys_Q'
    phys_action = 'phys_action'
    ai_action = 'ai_action'
    
    data = tag_data(data)
    data2 = tag_data(data2)
    
    res_quan = diff_vs_outcome2(data, data2)
    return res_quan

    
    
    
    