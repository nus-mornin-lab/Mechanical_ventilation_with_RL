使用方式：
1. 下载最新处理过的数据，在 new/data/0523/ 目录下解压，并命名为：
	data_rl_240min_mimic.csv
	data_rl_60min_mimic.csv
	data_rl_240min_eicu.csv
	data_rl_60min_eicu.csv
2. 在 new/BCQ/setting.py 中更改setting_paras 里的内容 (标注“# change here for grid search”的地方)，来决定搜索的参数值。
	可调参数：TRAIN_SET, CUT_TIME, STEP_LENGTH, MISSING_CUT, SEED, a, b, c, BCQ_THRESHOLD, GAMMA, PRETRAIN
3. 在 new/BCQ/ 中运行 python run_Pendulum_BCQ.py
4. 在 new/result/ 中检查网格搜索的结果。stats.txt的第一行列出了搜索了哪些参数，从第二行开始的每一行是一个参数setting下的最佳结果。
	best score的最高位代表长期结果呈现V shape的数量，后两位代表短期结果呈现倒立V shape的数量，由于同时考虑了inner test 和 outter test 上的效果，所以理想的best_score是612。
	round表示这个结果出现在训练的第几轮。
5. 需进一步验证时，选择4中某个想要进一步验证的结果，更改new/BCQ/test_saved_model.py中的 setting_keys、setting_vals、setting_dir 三个变量值，然后运行 python test_saved_model.py，详细结果会输出在 new/result/ 中的一个新文件夹里。
	setting_keys：4中stats.txt 里第一行 “-- ”后的字符串；
	setting_vals：4中stats.txt 里想要进一步验证的结果对应行 “-- ”后的字符串；
	setting_dir：4中stats.txt 所在文件夹名

** eicu数据：https://drive.google.com/drive/folders/12RQKb0qffh_tIbPxNWDWY8QY4k_mcIYX?usp=sharing。
** mimic数据：https://drive.google.com/drive/folders/1dV42k_okbRJ8DdBvHwx90idzJIenuQFC?usp=sharing。
** 对于不需要搜索的参数值定义在 setting.py顶部 和 run_Pendulum_BCQ.py的regular_parameters中。
** 在setting.py顶部 和 run_Pendulum_BCQ.py的regular_parameters中 定义的参数值会在网格搜索时被设置搜索的参数值覆盖。




####### 0523之前的解释，不用看 ###########

训练方式：
mimic数据分为了5份：3份组成训练集，1份验证集，1份测试集。

（VALIDATION为False时） -- 默认
1. 使用预设超参数，在{训练集∪验证集}上训练模型。
2. 在{训练集∪验证集}上评估模型。
3. 在mimic数据的测试集上评估模型。
4. 在eicu数据上评估模型。

（VALIDATION为True时）
1. 训练集上训练使用不同的超参数组合训练，取验证集上的cwpdis value最大的结果对应的超参数作为最优超参数。
2. 使用最优超参数，在{训练集∪验证集}上训练最优模型。
3. 在{训练集∪验证集}上评估模型。
4. 在mimic数据的测试集上评估模型。
5. 在eicu数据上评估模型。

* 本地运行 60min的STEP_LENGTH 可能会报内存错误。
* 以 60min的STEP_LENGTH 时间窗划分数据后，数据长度过长，实际可能不适合用cwpdis来进行off-policy评估