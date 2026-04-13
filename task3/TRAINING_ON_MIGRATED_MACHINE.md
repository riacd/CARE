# Task 3 迁移机器环境启动指南

这份文档只讲两件事：

1. 如何解包 `runtime` 中的环境包
2. 如何正确激活这个环境并启动 Task 3 训练

## 1. 需要随项目一起迁移的内容

至少保留这些目录和文件：

```bash
baseline/CARE/runtime/CARE_CREEP_env.tar.gz
baseline/CARE/runtime/unpack_creep_env.sh
baseline/CARE/task3/data/
baseline/CARE/CREEP/data/pretrained_ProtT5/
baseline/CARE/CREEP/data/pretrained_SciBert/
baseline/CARE/CREEP/data/pretrained_rxnfp/
```

## 2. unpack 环境

进入 `runtime` 目录后执行：

```bash
cd baseline/CARE/runtime
bash unpack_creep_env.sh ./envs/CREEP
```

这一步会做两件事：

1. 把 `CARE_CREEP_env.tar.gz` 解压到 `./envs/CREEP`
2. 生成 `./envs/CREEP/activate_runtime.sh`

如果你想换解压目录，也可以这样：

```bash
cd baseline/CARE/runtime
bash unpack_creep_env.sh /your/path/to/CREEP
```

## 3. activate 环境

不要使用：

```bash
source ./envs/CREEP/bin/activate
```

要使用：

```bash
source ./envs/CREEP/activate_runtime.sh
```

原因是这个打包环境需要显式设置 `PYTHONHOME`。  
`activate_runtime.sh` 会自动根据脚本自身位置设置：

- `CARE_CREEP_ENV_DIR`
- `PYTHONHOME`
- `PATH`

所以它可以在任意目录下被 `source`：

```bash
source /absolute/path/to/baseline/CARE/runtime/envs/CREEP/activate_runtime.sh
```

## 4. 验证环境是否激活成功

激活后先检查 Python：

```bash
python -c "import sys; print(sys.executable); print(sys.prefix)"
```

正常情况下，输出应该指向你刚刚解包的 `runtime/envs/CREEP`。

再检查核心依赖：

```bash
python -c "import torch, transformers, sentencepiece; print(torch.__version__); print(transformers.__version__); print(sentencepiece.__version__)"
```

## 5. 加载 CUDA

如果目标机器使用 `module` 管理 CUDA，先加载 CUDA：

```bash
module load cuda/12.8
```

检查 GPU 和 PyTorch：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

如果 `torch.cuda.is_available()` 输出 `True`，就可以开始训练。

## 6. 启动 Task 3 训练

先回到 `baseline/CARE` 目录：

```bash
cd baseline/CARE
```

### enzyme split

```bash
python task3/CREEP/step_01_train_CREEP_task3.py \
  --device 0 \
  --split_type enzyme_split \
  --output_model_dir task3/output/enzyme_split_run \
  --wandb_mode offline
```

### rxn_sub_split

```bash
python task3/CREEP/step_01_train_CREEP_task3.py \
  --device 0 \
  --split_type rxn_sub_split \
  --output_model_dir task3/output/rxn_sub_split_run \
  --wandb_mode offline
```

## 7. 一条龙命令

如果你想在迁移后的机器上从零开始直接跑，可以按这个顺序：

```bash
cd baseline/CARE/runtime
bash unpack_creep_env.sh ./envs/CREEP
source ./envs/CREEP/activate_runtime.sh
module load cuda/12.8
cd ..
python task3/CREEP/step_01_train_CREEP_task3.py \
  --device 0 \
  --split_type enzyme_split \
  --output_model_dir task3/output/enzyme_split_run \
  --wandb_mode offline
```

## 8. smoke test

如果只想验证训练链路是否通，可以先跑一个极小配置：

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

## 9. 去 AAM 中间结果

Task 3 第一次训练时，会对 `all_pair_data.tsv` 中的 `mapped_rxn` 执行去 AAM 处理，并把中间结果单独保存下来。

默认保存目录是：

```bash
runtime/task3_preprocessed/
```

保存的内容是去 AAM 后的反应数据文件，不是整份训练集缓存。  
后续训练如果发现这个中间结果文件已经存在，就会直接加载，不再重复对整份 `all_pair_data.tsv` 做去 AAM。

如需改目录，可显式传参：

```bash
--preprocessed_rxn_dir /your/dir
```

## 10. 常见问题

### 10.1 `No module named 'site'`

一般说明你没有用 `activate_runtime.sh`，而是直接调用了包内 Python，或者用了 `bin/activate`。

正确做法：

```bash
source ./envs/CREEP/activate_runtime.sh
```

### 10.2 `torch.cuda.is_available()` 是 `False`

优先检查：

```bash
module load cuda/12.8
nvidia-smi
```

然后再确认当前 `python` 确实来自解包后的环境。

### 10.3 启动很慢

Task 3 启动时会先读取 `all_pair_data.tsv`，并对反应做一次预处理。全量数据下，启动慢是正常现象。
