# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:56:13 2020

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


def run_eval(data, loss, datatype, SEED = setting.SEED):

    def plot_loss(loss):
        if loss:
            plt.figure(figsize=(7,4))
            plt.plot(loss)
            plt.savefig(res_dir + 'loss.jpg',dpi = 100)
        
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
    
    def action_distribution3(data):
        data = data.reset_index(drop = True).copy()
        
        # overall
        plt.figure(figsize=(7,4))
        for i,v in enumerate(action_types):
            phys_col = v + '_level'
            ai_col = v + '_level_ai'    
            x_s_ = sorted(data[phys_col].unique())
            aa = data[phys_col].value_counts()[x_s_]
            bb = data[ai_col].value_counts()[x_s_]
            plt.subplot(131+i)
            index=np.arange(len(aa.index))
            plt.bar(
                index-1/8,
                aa,
                color='blue',
                width=1/4)
            plt.bar(
                index+1/8,
                bb,
                color='red',
                width=1/4)
            plt.xlabel(v+' action taken')
            if i == 0:
                plt.ylabel('Counts')
            if len(x_s_) == 3:
                plt.xticks(x_s_,['Low','Med','High'],size = 9)
            else:
                plt.xticks(x_s_,['Low','High'],size = 9)
        
        plt.legend(['Phys policy','AI policy'])
        # plt.title('Action counts of Phys & AI policy')
        plt.tight_layout()
        plt.savefig(res_dir + 'overall_action_distribution3.jpg',dpi = 200)
    
        # stratified action distribution   
        plt.figure(figsize=(8,7))
        stra_list = sorted(data[stratify_col].unique())
        for stra in stra_list:
            cur_data = data[data[stratify_col] == stra].copy()
            for i,v in enumerate(action_types):
                phys_col = v + '_level'
                ai_col = v + '_level_ai'    
                x_s_ = sorted(cur_data[phys_col].unique())
                aa = cur_data[phys_col].value_counts()[x_s_]
                bb = cur_data[ai_col].value_counts()[x_s_]
                plt.subplot(len(stra_list)*100+31+i+stra*3)
                index=np.arange(len(aa.index))
                plt.bar(
                    index-1/8,
                    aa,
                    color='blue',
                    width=1/4)
                plt.bar(
                    index+1/8,
                    bb,
                    color='red',
                    width=1/4)
                plt.xlabel(v+' action taken')
                if i == 0:
                    plt.ylabel('Counts  -  ' + stratify_col + str(int(stra)))
                if len(x_s_) == 3:
                    plt.xticks(x_s_,['Low','Med','High'],size = 9)
                else:
                    plt.xticks(x_s_,['Low','High'],size = 9)
            
        plt.legend(['Phys policy','AI policy'])
        # plt.title('Action counts of Phys & AI policy')
        plt.tight_layout()
        plt.savefig(res_dir + 'stratified_action_distribution3.jpg',dpi = 200)        
    
    def diff_vs_outcome(data, outcome_col1 = 'hosp_mort',outcome_col2 = 'spo2_reach', bootstrap_round = 500):
    
    # vs_motality
        data = data.reset_index(drop = True).copy()
        # 4-hour level 
        x_s = {}
        res = {}
        res_ = {}
        for v in action_types:
            x_s[v] = sorted(set(data[v + '_diff']))
            res[v] = {}
            res_[v] = {}
            for k in x_s[v]:
                res[v][k] = []
                res_[v][k] = []
            
        used_data = data[[v + '_diff' for v in action_types] + [outcome_col1, outcome_col2 ]]
        for i in range(bootstrap_round):
            if i%10 == 0:
                print ('bootstrap 4-hour level: ' + str(i) + '...')

            df_index = np.random.choice(used_data.index, size = len(used_data))
            # visit level
            # diff vs motality
            for v in action_types:
                # v = 'PEEP'
                diff_col = v + '_diff'
                # aa = used_data.loc[df_index].groupby(diff_col).apply(lambda dt: dt[outcome_col1].mean())
                aa1 = used_data.loc[df_index].groupby(diff_col).apply(lambda dt: dt[[outcome_col1,outcome_col2]].mean())
                for k in x_s[v]:
                    res[v][k].append(aa1.loc[k,outcome_col1])
                    res_[v][k].append(aa1.loc[k,outcome_col2])
            
        # patient level 
        x_s_a = {}
        res_a = {}
        x_s_b = {}
        res_b = {}
        for v in action_types:
            x_s_a[v] = sorted(set(data[v + '_diff_mean_level']))
            res_a[v] = {}
            x_s_b[v] = sorted(set(data[v + '_conc_rate_level']))
            res_b[v] = {}
            for k in x_s_a[v]:
                res_a[v][k] = []
            for k in x_s_b[v]:
                res_b[v][k] = []

        used_data = data[[v + '_diff_mean_level' for v in action_types] + [v + '_conc_rate_level' for v in action_types] + ['patientunitstayid',outcome_col1 ]].drop_duplicates(subset = ['patientunitstayid'])
        for i in range(bootstrap_round):
            if i%10 == 0:
                print ('bootstrap: patient level: ' + str(i) + '...')

            df_index = np.random.choice(used_data.index, size = len(used_data))
            # patient level
            for v in action_types:
                diff_mean_col = v + '_diff_mean_level'
                conc_rate_col = v + '_conc_rate_level'
                aa = used_data.loc[df_index].groupby(diff_mean_col).apply(lambda dt: dt[outcome_col1].mean())
                bb = used_data.loc[df_index].groupby(conc_rate_col).apply(lambda dt: dt[outcome_col1].mean())
                for k in x_s_a[v]:
                    if k in aa.index:
                        res_a[v][k].append(aa[k])  
                for k in x_s_b[v]:
                    if k in bb.index:
                        res_b[v][k].append(bb[k])  
     
        
        # plot 
        def plot_vs_outcome(x_s, res, x_name, y_name, title_name,color_):
            res_025 = {}
            res_500 = {}
            res_975 = {}
            for v in action_types:
                res_025[v] = []
                res_500[v] = []
                res_975[v] = []
                for k in (x_s[v]):
                    cur = np.percentile(res[v][k], [2.5, 50, 97.5]) 
                    res_025[v].append(cur[0])
                    res_500[v].append(cur[1])
                    res_975[v].append(cur[2])
            plt.figure(figsize=(12,4))
            for i,v in enumerate(action_types):
                plt.subplot(131+i)
                line1, = plt.plot(x_s[v],res_500[v], color=color_, lw=1.5, ls='-', ms=4)
                plt.fill_between(x_s[v],res_025[v], res_975[v], color=color_, alpha=0.6)
                if x_name != 'Concordant rate':
                    plt.xticks(x_s[v])
                else:
                    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                plt.grid()
                plt.xlabel(x_name)
                if i == 0:
                    plt.ylabel(y_name)
                plt.title(v)
            
            plt.savefig(res_dir + title_name +'.jpg',dpi = 200)

        plot_vs_outcome(x_s, res, 'Model action - Phys action', 'Motality', 'diff4h_vs_motality', 'red')
        plot_vs_outcome(x_s_a, res_a, 'Model action - Phys action','Motality', 'diffPatientlevel_vs_motality','brown')
        plot_vs_outcome(x_s_b, res_b, 'Concordant rate','Motality', 'conc_vs_motality','green')
        
        plot_vs_outcome(x_s, res_, 'Model action - Phys action','spo2_reach', 'diff4h_vs_spo2reach', 'pink')


    def q_vs_outcome(data, outcome_col1 = 'hosp_mort', outcome_col2 = 'spo2_reach'):
        # Q_vs_motality    
        # data[phys_Q] = [random.uniform(-10,10) for i in range(len(data))]
        data = data.reset_index(drop = True).copy()    
        
        data[phys_Q + '_discre'] = data[phys_Q].apply(lambda x: round(x,2))
        Q_s = sorted(set(data[phys_Q + '_discre']))
        
        # q vs motality #####################
        bb = data.groupby(phys_Q + '_discre').apply(lambda dt: [dt[outcome_col1].mean(),dt[outcome_col1].sem()])
    
        res_025_Q = []
        res_500_Q = []
        res_975_Q = []
        for k in Q_s:
            res_500_Q.append(bb[k][0])
            res_025_Q.append(bb[k][0] - bb[k][1])
            res_975_Q.append(bb[k][0] + bb[k][1])
        plt.figure(figsize=(6,4))
        line1, = plt.plot(Q_s,res_500_Q, color='blue', lw=1.5, ls='-', ms=4)
        plt.fill_between(Q_s,res_025_Q, res_975_Q, color='blue', alpha=0.6)
        # plt.xticks(Q_s)
        # plt.grid()
        plt.xlabel('Return of actions')
        plt.ylabel('Motality risk')
        
        plt.savefig(res_dir + 'q_vs_motality.jpg',dpi = 200)
        
        res_dt = pd.DataFrame()
        res_dt['bb'] = bb
        res_dt['dr'] = res_dt['bb'].apply(lambda x: str(round(x[0]*100,1)) + '+-' + str(round(x[1]*100,1)) + '%')
        
        # q vs sop2_reach #####################
        bb = data.groupby(phys_Q + '_discre').apply(lambda dt: [dt[outcome_col2].mean(),dt[outcome_col2].sem()])
        
        res_025_Q = []
        res_500_Q = []
        res_975_Q = []
        for k in Q_s:
            res_500_Q.append(bb[k][0])
            res_025_Q.append(bb[k][0] - bb[k][1])
            res_975_Q.append(bb[k][0] + bb[k][1])
        plt.figure(figsize=(6,4))
        line1, = plt.plot(Q_s,res_500_Q, color='lightblue', lw=1.5, ls='-', ms=4)
        plt.fill_between(Q_s,res_025_Q, res_975_Q, color='lightblue', alpha=0.6)
        # plt.xticks(Q_s)
        # plt.grid()
        plt.xlabel('Return of actions')
        plt.ylabel('sop2 reach prob')
        
        plt.savefig(res_dir + 'q_vs_spo2.jpg',dpi = 200)
        
        res_dt_ = pd.DataFrame()
        res_dt_['bb'] = bb
        res_dt_['rr'] = res_dt_['bb'].apply(lambda x: str(round(x[0]*100,1)) + '+-' + str(round(x[1]*100,1)) + '%')
        
        return res_dt, res_dt_
    
    def quantitive_eval(data,res_dt,res_dt_, outcome_col = 'hosp_mort'):
        data['Random_action'] = [random.randrange(action_num) for i in range(len(data))]
        cc = data[phys_action].value_counts()
        data['One-size-fit-all_action'] = cc.index[np.argmax(cc)]
        
        data['Random_Q'] = data.apply(lambda x: x['Q_' + str(int(x['Random_action']))],axis = 1)
        data['One-size-fit-all_Q'] = data.apply(lambda x: x['Q_' + str(int(x['One-size-fit-all_action']))],axis = 1)
        
        # maybe change to doubly robust estimation / WIS
        q_dr_dt = pd.DataFrame()
        for mod in ['ai','phys','Random','One-size-fit-all']:    
            q_dr_dt.loc[mod,'Q'] = data[mod + '_Q'].mean()
    
        def find_nearest_Q(Q_mean,res_dt):
            ind = np.argmin([abs(Q_mean - i) for i in res_dt.index])
            Q_res = res_dt.index[ind]
            return Q_res
        
        for mod in ['ai','phys','Random','One-size-fit-all']:
            q_dr_dt.loc[mod,'deathrate'] = res_dt.loc[find_nearest_Q(q_dr_dt.loc[mod,'Q'] , res_dt),'dr']
            q_dr_dt.loc[mod,'spo2reachrate'] = res_dt_.loc[find_nearest_Q(q_dr_dt.loc[mod,'Q'], res_dt_),'rr']
        q_dr_dt.to_csv(res_dir + 'qmean_and_deathreachrate.csv', encoding = 'gb18030')
            
        return q_dr_dt
    
    def action_concordant_rate(data):
        conc_dt = pd.DataFrame()
        for i,v in enumerate(action_types):
            phys_col = v + '_level'
            ai_col = v + '_level_ai' 
            conc_dt.loc[v, 'concordant_rate'] = str(round(np.mean(data[phys_col] == data[ai_col])*100,1)) + '%'
            
        conc_dt.loc['all', 'concordant_rate'] = str(round(np.mean(data[phys_action] == data[ai_action])*100,1)) + '%'
        
        conc_dt.to_csv(res_dir + 'action_concordant_rate.csv', encoding = 'gb18030')
        
        return conc_dt
    
    np.random.seed(523)
    random.seed(523)
    action_num = 18
    Q_list = ['Q_' + str(i) for i in range(action_num)]
    action_types = ['PEEP', 'FiO2', 'Tidal']
    phys_Q = 'phys_Q'
    phys_action = 'phys_action'
    ai_action = 'ai_action'
    
    data['spo2_reach'] = data['next_ori_spo2'].apply(lambda x: (x >= 94 and x <= 98)+0 )
    if phys_action not in data.columns.tolist():
        data[phys_action] = data.apply(lambda x: int(x[action_types[0]+'_level']*9 + x[action_types[1]+'_level']*3 + x[action_types[2]+'_level']),axis = 1)
    data['ai_Q'] = np.max(data[Q_list],axis = 1)
    data[ai_action] = np.argmax(np.array(data[Q_list]),axis = 1)
    data[phys_Q] = data.apply(lambda x: x['Q_' + str(int(x[phys_action]))], axis = 1)
    
    data[action_types[0] + '_level_ai'] = (data[ai_action]/9).apply(lambda x: int(x))
    data[action_types[1] + '_level_ai'] = (data[ai_action]%9/3).apply(lambda x: int(x))
    data[action_types[2] + '_level_ai'] = (data[ai_action]%9%3).apply(lambda x: int(x))
    
    
    for v in action_types:
        data[v + '_diff'] = data[v + '_level'] - data[v + '_level_ai']
        data[v + '_conc'] = (data[v + '_level'] == data[v + '_level_ai']) + 0
        
    data = data.groupby('patientunitstayid').apply(tag_conc_rate_and_diff_mean)

    for v in action_types:    
        data[v + '_diff_mean_level'] = data[v + '_diff_mean'].apply(discre_diff_level)
        data[v + '_conc_rate_level'] = data[v + '_conc_rate'].apply(discre_conc_level)

    data = data.reset_index(drop = True).copy()
    
    stratify_col = 'SOFA_level'
    data[stratify_col] = (data['ori_sofatotal'] <= 6)+0 
    
    res_dir = 'result/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + setting.REWARD_FUN + '_' + datatype +'_' + str(SEED) + '_' + setting.MODEL + '/'
    if os.path.isdir(res_dir) == False:
        os.makedirs(res_dir)            
    action_distribution3(data)
    diff_vs_outcome(data)
    res_dt,res_dt_ = q_vs_outcome(data)
    q_dr_dt = quantitive_eval(data, res_dt,res_dt_) 
    conc_dt = action_concordant_rate(data) 
    plot_loss(loss)
    print (q_dr_dt)
    print (conc_dt)
    