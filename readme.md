1. 运行方法
	- ```python run_Pendulum.py -m train -d mimic``` / ```python run_Pendulum.py -m train -d eicu```
	- ```python run_Pendulum.py --mode train --data mimic``` / ```python run_Pendulum.py --mode train --data eicu```
	- mode可取值：train/eval，但保存和读取模型的code还没完全写好
	- data代表数据：可取值为mimic/eicu
	- first_run先默认值为True了，这个参数设置不是很必要

2. 交叉测试
	- 新增参数: -e / --eval_model：选择用于评估的模型（对应的训练数据）
	- train on eicu and eval on mimic
    	- 训练模型：```python run_Pendulum.py -m train -d eicu```
    	- ```python run_Pendulum.py -m eval -d mimic -e eicu```
  	- train on mimic and eval on eicu
    	- 训练模型：```python run_Pendulum.py -m train -d mimic```
    	- ```python run_Pendulum.py -m eval -d eicu -e mimic```