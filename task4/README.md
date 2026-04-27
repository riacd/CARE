# task4 任务说明
task 4 需要测试模型的挖酶性能，benchmark 与模型设计方式如下
## benchmark
benchmark 设计如下


## 数据
数据有两种切分，分别是按酶切分和按反应底物切粉，分别位于 
1. @/data/enzyme_split
2. @/data/rxn_sub_split
包含 train & val 的 tsv 文件，文件中只描述 rxn & enz 的索引，数据需要根据索引在数据库中提取
1. 蛋白数据库：@/data/pair_merged_data/enzyme_db_extended.json
2. 反应数据库：反应数据和索引的对应关系需要从 @/data/pair_merged_data/all_pair_data.tsv 中提取

为保证 当前项目 可直接整体迁移到其它机器，Task 3 训练脚本默认只依赖项目目录内目录内的文件：
1. `/data/*` 中的 split 与 pair 数据
2. `/CREEP/data/pretrained_*` 中的初始化模型权重缓存


## 模型
模型使用 CREEP ，需要在此基础上
1. 更新模型设计
2. 更新训练代码
3. 更新推理代码

### 模态
使用 三模态 对比学习
1. protein 模态
2. reaction 模态
3. 文本 模态

#### 文本数据收集
已有的数据都是 protein, reaction 的 pair-wise 数据
每一段文本模态内容如下：EC text + '\t' + IUPAC text
#### IUPAC text
需要使用如下步骤：
1. 从 @data/pair_merged_data/all_pair_data.tsv 中读取每个 reaction 的 rxn_id 和 SMILES 表达式
2. 将每个反应表达式拆分成其反应物 SMILES 集合以及产物 SMILES 集合。
3. 对所有反应的反应物和产物分子取交集，得到一个所有分子SMILES，用文件保存，并根据 SMILES 找到每个分子对应的 IUPAC 名称，用一个 json 保存映射关系
4. 再根据每个反应的SMILES 表达式重新获得 反应的 IUPAC text，具体方式为把原本的 反应SMILES 中每个分子的 SMILES 替换成 其对应的 IUPAC 名称，将SMILES反应中的 '.' 替换为 ‘ + ’， '>>' 替换为 '->'

#### EC text
文件描述如下：
1. @processed_data/text2EC.csv 描述了 EC number 与 EC text 的对应关系

获取每个反应对应的 EC text 步骤如下：
1. 从 @data/pair_merged_data/all_pair_data.tsv 中获得全部 (rxn_id, enz_id) pair
2. 根据 @data/proteins_afdb.tsv 中描述的 enz_id -> EC number 映射，可能一个 enz 对应多个 EC number，此时需要扩展，获得全部 (rxn_id, enz_id, EC number) ，其中每条只包含一个 EC number 【如果有enz_id 没有对应的 EC number 映射，则删除该条】
3. 现在计算每个 rxn_id 对应的 EC number，具体算法为包含这个反应的所有 (rxn_id, enz_id, EC number) 条目中，最多出现的 EC number 就是反应对应的 EC number，如果有相同数目取前面那个
4. 得到每个 rxn_id 的 EC number 以后，根据 @processed_data/text2EC.csv 中得到 EC number 对应的 EC text 
5. 对于没有对应 EC text 的 rxn_id，它们的 EC text 取 ''

### 模型设计
使用 @/CREEP/CREEP/models 中的 CREEP 模型，模型包含
1. ProtT5：蛋白编码器，输入蛋白序列
2. BERT：反应编码器用于提取反应的 rxnfp，输入没有经过mapped 的反应SMILES序列

### 训练
#### 训练方式
训练方式和 @task2_baselines/CREEP/step_01_train_CREEP.py 中的训练方法保持一致（只使用蛋白-反应两个模态的训练）
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


## 打包
运行项目的环境被打包在目录 @runtime 之下

迁移到新机器后的启动方式见：
- `task3/TRAINING_ON_MIGRATED_MACHINE.md`
