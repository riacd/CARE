# Task 3 迁移机器训练说明

这份文档说明如何在迁移后的机器上，使用仓库内自带的打包环境启动 Task 3 训练。

## 1. 前提

需要保证以下目录已经随项目一起迁移：

- `baseline/CARE/task3/data/`
- `baseline/CARE/CREEP/data/pretrained_ProtT5/`
- `baseline/CARE/CREEP/data/pretrained_SciBert/`
- `baseline/CARE/CREEP/data/pretrained_rxnfp/`
- `baseline/CARE/runtime/CARE_CREEP_env.tar.gz`

Task 3 训练脚本默认从这些目录读取数据和预训练权重缓存，不依赖外部下载。

## 2. 解包运行环境

在项目根目录下执行：

```bash
cd baseline/CARE/runtime
bash unpack_creep_env.sh ./envs/CREEP
source ./envs/CREEP/activate_runtime.sh
cd ..
```

注意：

- 不要直接使用 `./envs/CREEP/bin/activate`
- 这里必须使用 `activate_runtime.sh`
- 该脚本会根据自身位置设置 `PYTHONHOME`，否则包内 Python 可能无法正常启动
- 因此也可以在任意目录执行 `source /absolute/path/to/activate_runtime.sh`

## 3. 加载 CUDA 模块

如果目标机器通过 `module` 管理 CUDA，先加载 CUDA。当前验证通过的命令是：

```bash
module load cuda/12.8
```

快速检查 GPU 与 PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

如果这里输出 `True` 且能看到 GPU 名称，就可以继续训练。

## 4. 启动训练

### enzyme split

```bash
python task3/CREEP/step_01_train_CREEP_task3.py \
  --device 0 \
  --split_type enzyme_split \
  --output_model_dir task3/output/enzyme_split_run \
  --wandb_mode offline
```

### reaction-substrate split

```bash
python task3/CREEP/step_01_train_CREEP_task3.py \
  --device 0 \
  --split_type rxn_sub_split \
  --output_model_dir task3/output/rxn_sub_split_run \
  --wandb_mode offline
```

脚本会自动使用默认数据路径：

- `task3/data/enzyme_split/train_pairs.tsv`
- `task3/data/enzyme_split/val_pairs.tsv`
- `task3/data/rxn_sub_split/train_reactions.tsv`
- `task3/data/rxn_sub_split/val_reactions.tsv`
- `task3/data/pair_merged_data/all_pair_data.tsv`
- `task3/data/pair_merged_data/enzyme_db_extended.json`

## 5. 常用参数

可以按需覆盖以下参数：

```bash
--epochs 40
--batch_size 16
--num_batches_per_epoch 5000
--val_num_batches 500
--num_workers 0
--wandb_mode offline
```

如果只想做快速 smoke test，可以这样跑：

```bash
python task3/CREEP/step_01_train_CREEP_task3.py \
  --device 0 \
  --split_type enzyme_split \
  --epochs 1 \
  --batch_size 2 \
  --num_batches_per_epoch 1 \
  --val_num_batches 1 \
  --num_workers 0 \
  --output_model_dir runtime/task3_smoke \
  --wandb_mode disabled
```

## 6. 训练输出

训练输出目录下通常会包含：

- `log.txt`
- `train_dataset_stats.json`
- `val_dataset_stats.json`
- `*_model.pth`
- `*_model_final.pth`

其中：

- `model.pth` 系列对应当前最优验证结果
- `model_final.pth` 系列对应最后一个 epoch

## 7. 已验证的最小运行方式

以下组合已经在当前仓库中通过：

```bash
source runtime/envs/CREEP/activate_runtime.sh
module load cuda/12.8
python task3/CREEP/step_01_train_CREEP_task3.py \
  --device 0 \
  --epochs 1 \
  --batch_size 2 \
  --num_batches_per_epoch 1 \
  --val_num_batches 1 \
  --num_workers 0 \
  --output_model_dir runtime/task3_smoke_subset \
  --wandb_mode disabled
```

该验证已经完成：

- 模型加载
- 数据集构建
- train 一轮
- val 一轮
- checkpoint 保存

## 8. 常见问题

### 8.1 `No module named 'site'`

通常说明没有通过 `activate_runtime.sh` 激活包内环境，或者 `PYTHONHOME` 没有设置。

重新执行：

```bash
source runtime/envs/CREEP/activate_runtime.sh
```

### 8.2 `torch.cuda.is_available()` 为 `False`

优先检查：

```bash
module load cuda/12.8
nvidia-smi
```

如果 `nvidia-smi` 正常，但 PyTorch 仍然看不到卡，通常是 CUDA 模块没有加载到当前 shell，或者在错误的环境里运行了 Python。

### 8.3 启动很慢

这是正常现象。Task 3 在训练开始前会读取 `all_pair_data.tsv`，并对反应做一次去 mapping 预处理。全量数据下，启动时间会明显长于单步训练本身。

### 8.4 RDKit 反应合法性 warning

如果看到少量类似 `Explicit valence ... is greater than permitted` 的 warning，通常是个别反应在去 mapping 时无法被 RDKit 正常解析。脚本会跳过无法使用的样本，并在 `train_dataset_stats.json` / `val_dataset_stats.json` 中体现统计结果。
