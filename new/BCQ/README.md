运行方式：
1. 下载最新处理过的数据，放在上一级目录下解压。地址：https://drive.google.com/drive/folders/1T5kk1JqSfudEFi6vhECeQkn1Y7SFsAiW
2. 运行python run_Pendulum_BCQ.py

在setting.py中设置VALIDATION为True可利用验证集选择超参数。

训练方式：
mimic数据分为了5份：3份组成训练集，1份验证集，1份测试集。

（VALIDATION为True时）
1. 训练集上训练使用不同的超参数组合训练，取验证集上的cwpdis value最大的结果对应的超参数作为最优超参数。
2. 使用最优超参数，在{训练集∪验证集}上训练最优模型。
3. 在{训练集∪验证集}上评估模型。
4. 在mimic数据的测试集上评估模型。
5. 在eicu数据上评估模型。

（VALIDATION为False时）
1. 使用预设超参数，在{训练集∪验证集}上训练模型。
2. 在{训练集∪验证集}上评估模型。
3. 在mimic数据的测试集上评估模型。
4. 在eicu数据上评估模型。

* 本地运行可能会报内存错误。
* 由于time window长度变成了原来的约1/4，reward中短期reward调成了原来的1/4，discount factor默认值0.99
* 以1hour时间窗划分数据后，数据长度过长，实际可能不适合用cwpdis来进行off-policy评估