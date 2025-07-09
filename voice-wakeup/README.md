# 语音唤醒系统 (Voice Wake-up System)

基于PyTorch的实时语音唤醒检测系统，采用CRNN（卷积循环神经网络）架构，实现高精度的唤醒词识别。

## 项目概述

本项目实现了一个完整的语音唤醒系统，包含数据采集、预处理、模型训练、评估测试和实时检测功能。系统能够在嘈杂环境中准确识别特定的唤醒词，并提供实时响应。

### 主要特性

- **高精度识别**：基于CRNN架构，结合CNN的特征提取和RNN的时序建模能力
- **实时检测**：支持实时音频流处理，低延迟响应
- **鲁棒性强**：内置音频增强和降噪处理，适应不同环境
- **易于扩展**：模块化设计，支持自定义唤醒词和模型优化

## 项目结构

```
voice-wakeup/
├── audio_processor.py      # 音频处理模块
├── crnn.py                # CRNN模型定义
├── dataset.py             # 数据集处理
├── train.py               # 模型训练脚本
├── test.py                # 模型评估脚本
├── realtime_test.py       # 实时检测脚本
├── get_voice.py           # 语音数据采集
├── data_splite.py         # 数据集分割
├── dataset/               # 数据集目录
│   ├── train/            # 原始训练数据
│   │   ├── wake/         # 唤醒词样本
│   │   └── not_wake/     # 非唤醒词样本
│   └── split_dataset/    # 分割后的数据集
│       ├── train/        # 训练集
│       └── val/          # 验证集
├── checkpoints/           # 模型检查点
│   └── best_model.pth    # 最佳模型
├── results/              # 测试结果
│   ├── confusion_matrix.png  # 混淆矩阵
│   └── errors.txt        # 错误样本记录
└── README.md             # 项目说明文档
```

## 技术架构

### 模型架构：CRNN (Convolutional Recurrent Neural Network)

```
输入音频 → 特征提取 → CNN特征提取 → RNN时序建模 → 分类输出
    ↓         ↓           ↓            ↓          ↓
  原始音频    MFCC       局部特征      时序特征    唤醒词概率
```

#### 网络结构详解

1. **特征提取层**
   - 使用MFCC（梅尔频率倒谱系数）作为音频特征
   - 包含39维特征：13个MFCC系数 + 13个一阶差分 + 13个二阶差分
   - 时间窗口长度：100帧（约1秒）

2. **CNN特征提取层**
   - 第一层：Conv2d(1→32) + BatchNorm + GELU + MaxPool
   - 第二层：Conv2d(32→64) + BatchNorm + GELU + MaxPool
   - 自适应池化：AdaptiveAvgPool2d调整特征维度

3. **RNN时序建模层**
   - 双向GRU：2层，隐藏单元128
   - 输入维度：576（64通道 × 9频率特征）
   - 输出维度：256（128 × 2方向）

4. **分类层**
   - 全连接层：256→128→2
   - 激活函数：GELU
   - 正则化：Dropout(0.4)

### 音频处理流程

```
原始音频 → 预加重 → VAD检测 → MFCC提取 → 差分计算 → 标准化 → 长度对齐
```

## 环境要求

### 系统要求
- Python 3.7+
- 操作系统：macOS/Linux/Windows
- 内存：≥8GB
- 显卡：支持CUDA的GPU（推荐）

### 依赖包
```bash
torch>=1.9.0
torchaudio>=0.9.0
librosa>=0.8.0
sounddevice>=0.4.0
soundfile>=0.10.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
tqdm>=4.60.0
```

## 安装说明

### 1. 克隆项目
```bash
git clone <repository-url>
cd voice-wakeup
```

### 2. 安装依赖
```bash
pip install torch torchaudio librosa sounddevice soundfile numpy scikit-learn matplotlib tqdm
```

### 3. 验证安装
```bash
python -c "import torch; print(torch.__version__)"
python -c "import librosa; print(librosa.__version__)"
```

## 使用方法

### 1. 数据采集

使用`get_voice.py`采集训练数据：

```bash
python get_voice.py
```

**采集流程：**
- 唤醒词样本：需要清晰说出唤醒词，建议采集50-100个样本
- 背景音样本：录制环境音、日常对话等，建议采集150-300个样本

**采集注意事项：**
- 保持录音环境一致
- 唤醒词发音清晰
- 背景音多样化（安静、嘈杂、音乐等）

### 2. 数据集划分

使用`data_splite.py`将数据集划分为训练集和验证集：

```bash
python data_splite.py
```

- 训练集：80%
- 验证集：20%
- 随机种子：42（确保可复现）

### 3. 模型训练

使用`train.py`训练模型：

```bash
python train.py --epochs 100 --batch_size 40 --lr 1e-4
```

**训练参数：**
- `--train_dir`: 训练数据目录（默认：dataset/split_dataset/train）
- `--valid_dir`: 验证数据目录（默认：dataset/split_dataset/val）
- `--save_dir`: 模型保存目录（默认：./checkpoints）
- `--batch_size`: 批次大小（默认：40）
- `--epochs`: 训练轮数（默认：5000）
- `--lr`: 学习率（默认：1e-4）
- `--pos_weight`: 正样本权重（默认：3.0）
- `--device`: 设备选择（默认：自动检测）

**训练特性：**
- 自动保存最佳模型
- 学习率调度：OneCycleLR
- 梯度裁剪：防止梯度爆炸
- 类别平衡：加权损失函数
- 数据增强：音频增强技术

### 4. 模型评估

使用`test.py`评估模型性能：

```bash
python test.py --test_dir dataset/split_dataset/val --model_path checkpoints/best_model.pth
```

**评估指标：**
- 准确率、精确率、召回率、F1分数
- 混淆矩阵可视化
- 错误样本分析

### 5. 实时检测

使用`realtime_test.py`进行实时语音唤醒检测：

```bash
python realtime_test.py --model_path checkpoints/best_model.pth --threshold 0.85
```

**实时检测参数：**
- `--model_path`: 模型路径
- `--threshold`: 检测阈值（默认：0.85）
- `--device`: 设备选择

**检测特性：**
- 实时音频流处理
- 滑动窗口检测
- 连续触发确认
- 可调节敏感度

## 数据集说明

### 目录结构
```
dataset/
├── train/                 # 原始数据
│   ├── wake/              # 唤醒词样本（.wav格式）
│   └── not_wake/          # 非唤醒词样本（.wav格式）
└── split_dataset/         # 分割后数据
    ├── train/             # 训练集（80%）
    └── val/               # 验证集（20%）
```

### 音频格式要求
- 格式：WAV（16-bit PCM）
- 采样率：16000 Hz
- 通道数：单声道
- 时长：1-5秒（推荐3秒）

### 数据质量要求
- 唤醒词发音清晰
- 音频无明显截断
- 背景噪声适中
- 类别分布相对均衡

## 模型性能

### 训练配置
- 输入特征：39维MFCC + 差分特征
- 序列长度：100帧（约1秒）
- 批次大小：40
- 优化器：AdamW + OneCycleLR
- 正则化：Dropout + 权重衰减

### 性能指标
- 验证集准确率：>95%
- 假正率：<5%
- 假负率：<3%
- 实时检测延迟：<100ms

## 优化建议

### 数据层面
1. **增加数据多样性**：不同说话人、环境、音质
2. **数据平衡**：确保正负样本比例适当
3. **质量控制**：过滤异常或低质量样本

### 模型层面
1. **超参数调优**：学习率、批次大小、网络深度
2. **正则化技术**：Dropout、批归一化、权重衰减
3. **损失函数优化**：Focal Loss、AUC优化

### 部署层面
1. **模型量化**：减少模型大小和推理时间
2. **流水线优化**：音频预处理加速
3. **边缘计算**：支持移动设备部署

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   python train.py --batch_size 16
   ```

2. **音频格式错误**
   ```bash
   # 检查音频文件格式
   python -c "import librosa; print(librosa.get_samplerate('file.wav'))"
   ```

3. **实时检测无响应**
   ```bash
   # 检查音频设备
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

4. **模型加载失败**
   ```bash
   # 检查模型文件完整性
   python -c "import torch; torch.load('checkpoints/best_model.pth')"
   ```

### 性能调优

1. **提高准确率**
   - 增加训练数据量
   - 调整模型架构
   - 优化特征提取

2. **降低延迟**
   - 减少模型复杂度
   - 优化音频预处理
   - 使用模型加速技术

3. **增强鲁棒性**
   - 添加噪声增强
   - 多环境训练
   - 动态阈值调整

## 技术参考

### 核心算法
- MFCC特征提取
- 卷积神经网络（CNN）
- 循环神经网络（RNN/GRU）
- 语音活性检测（VAD）

### 相关论文
- "Attention-based Models for Speech Recognition"
- "Deep Learning for Wake Word Detection"
- "Convolutional Recurrent Neural Networks for Audio Classification"

### 开源项目
- PyTorch Audio
- LibROSA
- Mozilla DeepSpeech

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork本项目
2. 创建特性分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送到分支：`git push origin feature/new-feature`
5. 提交Pull Request

## 更新日志

### v1.0.0 (当前版本)
- 实现基础CRNN模型
- 支持实时语音检测
- 完善数据处理流程
- 添加评估和可视化工具

---

**如有问题，请提交Issue或联系项目维护者 vx:yiyan_k。** 