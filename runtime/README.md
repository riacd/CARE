# CARE Runtime

这个目录用于存放可迁移的运行环境包与环境清单。

当前约定：
- `CARE_CREEP_env.tar.gz`：可直接迁移的 `conda-pack` 环境包，用于运行 CARE 的 CREEP 相关训练与提取脚本
- `CARE_CREEP_env.yml`：环境导出清单
- `CARE_CREEP_env.explicit.txt`：conda 显式包列表
- `unpack_creep_env.sh`：目标机器上的解压与修复脚本

目标机器上的典型使用方式：

```bash
cd baseline/CARE/runtime
bash unpack_creep_env.sh ./envs/CREEP
source ./envs/CREEP/bin/activate
cd ..
```

如果目标机器没有兼容的 CUDA 运行时，仍可能只能走 CPU。
