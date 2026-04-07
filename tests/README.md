# Diff4Splat 测试脚本说明
========================

本目录包含 Diff4Splat 项目的测试脚本，方便快速验证环境和核心功能。

## 快速开始

### 1. 环境测试
首先验证环境是否配置正确：
```bash
python tests/test_environment.py
```

### 2. Wan 模型测试
测试 Wan 模型加载和 checkpoint 兼容性：
```bash
python tests/test_wan_model.py
```

### 3. 潜在对齐测试
测试 WanVAE ↔ TinyVAE 潜在空间对齐：
```bash
python tests/test_latent_alignment.py
```

## 测试文件说明

| 文件 | 功能 |
|------|------|
| `test_environment.py` | 验证 Python 环境、依赖包、checkpoints |
| `test_wan_model.py` | 测试 Wan 模型加载 + checkpoint 对比 |
| `test_latent_alignment.py` | 测试 WanVAE 和 TinyVAE 对齐 |
| `test_aether.py` | Aether 相关测试 |

## 项目结构总览

```
Diff4Splat/
├── src/                    # 主源码目录
│   ├── models/           # 模型定义
│   ├── data/             # 数据加载
│   └── utils/            # 工具函数
├── diff3r_src/           # Diff3R 相关代码
├── tests/                # ← 测试脚本（本目录）
├── configs/              # 配置文件
├── resources/            # 资源和 checkpoints
└── scripts/              # 训练脚本
```
