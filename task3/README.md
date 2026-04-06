# task3 任务说明
task 3 需要测试模型的挖酶性能，benchmark 与模型设计方式如下
## benchmark
benchmark 设计如下


## 数据
数据有两种切分，分别是按酶切分和按反应底物切粉，分别位于 
1. @baseline/CARE/task3/data/enzyme_split
2. @baseline/CARE/task3/data/rxn_sub_split
包含 train & val 的 tsv 文件，文件中只描述 rxn & enz 的索引，数据需要根据索引在数据库中提取
1. 蛋白数据库：@baseline/CARE/task3/data/pair_merged_data/enzyme_db_extended.json
2. 反应数据库：反应数据和索引的对应关系需要从 @baseline/CARE/task3/data/pair_merged_data/all_pair_data.tsv 中提取

为保证 `baseline/CARE` 可直接整体迁移到其它机器，Task 3 训练脚本默认只依赖 `baseline/CARE` 目录内的：
1. `baseline/CARE/task3/data/*` 中的 split 与 pair 数据
2. `baseline/CARE/CREEP/data/pretrained_*` 中的初始化模型权重缓存


## 模型
模型使用 CREEP ，需要在此基础上
1. 更新模型设计
2. 更新训练代码
3. 更新推理代码
### 模型设计
使用 @baseline/CARE/CREEP/CREEP/models 中的 CREEP 模型，模型包含
1. ProtT5：蛋白编码器，输入蛋白序列
2. BERT：反应编码器用于提取反应的 rxnfp，输入没有经过mapped 的反应SMILES序列

### 训练
#### 训练方式
训练方式和 @baseline/CARE/task2_baselines/CREEP/step_01_train_CREEP.py 中的训练方法保持一致（只使用蛋白-反应两个模态的训练）
但是 Batch 构造方式有所变化，原本使用 EC 分类抽取一个anchor pair 和 batch_size - 1 个 negative pair，现在使用
#### Batch 构造
Batch 构造方式实现具体请参考 @rxnzyme/data/samplers/prorxn.py 中的方式，接下来我详细描述如何从 train 的 reaction-enzyme pair 数据中构造训练batch
每个 epoch 包含 5000 个 batch
每个 batch 构造方式如下：
1. 从 train data 所有的 positive pair 中随机抽取一个，作为 anchor reaction-enzyme pair。它从 train data 中 mask 出去
接下来的目标是抽取 batch_size - 1 个 negative reaction-enzyme pair。且保证取的 batch_size 个所有 reaction-enzyme pair，跨 pair 组合的 reaction 和 enzyme 一定不出现在训练集中。
接下来是如何抽取这 batch_size - 1 个 negative reaction-enzyme pair
2. 在 train data 中，根据前一次抽取的 reaction-enzyme pair，寻找和该 reaction 成对的 enzyme，以及这些 enzyme 所对应的所有pair，将它们全部从 train data 中 mask 出去
3. 同理，在 train data 中，每次根据前一次抽取的 reaction-enzyme pair，寻找和该 enzyme 成对的 reaction，以及这些 reaction 所对应的所有pair，将它们全部从 train data 中 mask 出去
4. 从 train data 中没有被 mask 的 pair 中随机抽取一个，加入到选取的 negative reaction-enzyme pair 中
5. 重复第2步，每次mask 后能挑选出一个negative reaction-enzyme pair，直到 negative reaction-enzyme pair 数量达到 batch_size - 1
#### 额外要求
1. 使用 wandb 记录训练损失，本地记录，允许在登录节点同步至云端 (wandb API key: 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)
