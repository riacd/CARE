# task3 任务说明
task 3 需要测试模型的挖酶性能，benchmark 与模型设计方式如下
## benchmark
benchmark 设计如下

## 数据
数据有两种切分，分别是按酶切分和按反应底物切粉，分别位于 
1. @task3/data/enzyme_split
2. @task3/data/rxn_sub_split
包含 train & val 的 tsv 文件，文件中只描述 rxn & enz 的索引，数据需要根据索引在数据库中提取
1. 蛋白数据库：@/task3/data/pair_merged_data/enzyme_db_extended.json
2. 反应数据库：反应数据和索引的对应关系需要从 @/task3/data/pair_merged_data/all_pair_data.tsv 中提取

为保证 当前项目 可直接整体迁移到其它机器，Task 3 训练脚本默认只依赖项目目录内目录内的文件：
1. `/task3/data/*` 中的 split 与 pair 数据
2. `/CREEP/data/pretrained_*` 中的初始化模型权重缓存


## 模型
模型使用 CREEP ，需要在此基础上
1. 更新模型设计
2. 更新训练代码
3. 更新推理代码
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

### 推理
需要实现推理代码。推理计算方式如下：计算 protein_encoder & reaction_encoder 的 cosine similarity，作为对 reaction-enzyme pair 的 prediction。
#### 目标
要求推理结果是返回一个推理 pred 矩阵，每一行对应一个反应，每一列对应一个酶，每个位置对应预测分数
有如下两种数据切分
1. @task3/data/enzyme_split
2. @task3/data/rxn_sub_split
因此需要分开构造两种切分对应的不同的 preds
#### 数据
1. 两个测试集的 positive pair 数据，分别位于：@data/pair_merged_data/enzyme_split/val_pairs.tsv 和 @data/pair_merged_data/rxn_sub_split/val_reactions.tsv
2. 测试benchmark的组织：1. 每一行是一个reaction，顺序按的包含测试集 positive pairs 的tsv 文件中的反应顺序；2. 每一列是enz，顺序按照enz_db json文件中的蛋白顺序；3. val tsv 中包含所有 positive pair 点；

包含 train & val 的 tsv 文件，文件中只描述 rxn & enz 的索引，数据需要根据索引在数据库中提取
1. 蛋白数据库：@/task3/data/pair_merged_data/enzyme_db_extended.json
2. 反应数据库：反应数据和索引的对应关系需要从 @/task3/data/pair_merged_data/all_pair_data.tsv 中提取

#### 运行方式
按 enzyme split 生成完整 pred 矩阵：

```bash
python task3/CREEP/step_02_infer_CREEP_task3.py \
  --device 0 \
  --pretrained_folder task3/output/enzyme_split_run \
  --split_type enzyme_split \
  --output_dir task3/output/inference_preds \
  --verbose
```

按 rxn_sub split 生成完整 pred 矩阵：

```bash
python task3/CREEP/step_02_infer_CREEP_task3.py \
  --device 0 \
  --pretrained_folder task3/output/rxn_sub_split_run \
  --split_type rxn_sub_split \
  --output_dir task3/output/inference_preds \
  --verbose
```

输出文件包括：

1. `<split_type>_preds.npy`：完整预测矩阵，shape 为 `[num_rxns, num_enzymes]`
2. `<split_type>_row_rxn_ids.tsv`：矩阵行对应的 reaction 顺序
3. `<split_type>_col_enz_ids.tsv`：矩阵列对应的 enzyme 顺序
4. `<split_type>_metadata.json`：推理输入与输出的元信息

如需做小规模 smoke test，可额外传：

```bash
--max_reactions 8 --max_enzymes 16
```


## 打包
运行项目的环境被打包在目录 @runtime 之下

迁移到新机器后的启动方式见：
- `task3/TRAINING_ON_MIGRATED_MACHINE.md`
