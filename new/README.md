运行方式：
1. 下载最新处理过的数据，在 new/data 目录下解压。地址：https://drive.google.com/file/d/1LneFLHyB2QPQ8DXmwIr-iHLno-WYacKV/view?usp=sharing
2. 在 new/BCQ 中运行 python run_Pendulum_BCQ.py 或 在 new/DDQN 中运行 python run_Pendulum_DQN.py

*在setting.py中设置参数，注释“可调”的都是可以尝试的。注意做BCQ和DDQN的对比实验时分别修改两个文件夹中setting.py中的参数


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