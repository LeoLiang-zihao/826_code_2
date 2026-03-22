# Kaggle 胸部 X 光 (CXR) 分类竞赛研究报告

> 本文档综合整理了 Kaggle 及相关平台上主要的胸部 X 光分类/检测竞赛的获胜方案与最佳实践，
> 重点关注对 **ResNet50 Binary Classifier for Pleural Effusion Detection** 有指导意义的技术方案。

---

## 目录

1. [RSNA Pneumonia Detection Challenge (2018)](#1-rsna-pneumonia-detection-challenge-2018)
2. [CheXpert Competition (Stanford)](#2-chexpert-competition-stanford)
3. [SIIM-ACR Pneumothorax Segmentation (2019)](#3-siim-acr-pneumothorax-segmentation-2019)
4. [VinBigData Chest X-ray Abnormalities Detection (2021)](#4-vinbigdata-chest-x-ray-abnormalities-detection-2021)
5. [RANZCR CLiP Catheter and Line Position Challenge (2021)](#5-ranzcr-clip-catheter-and-line-position-challenge-2021)
6. [CheXNet / NIH ChestX-ray14 基准研究](#6-chexnet--nih-chestx-ray14-基准研究)
7. [图像分辨率对 CXR 分类的影响](#7-图像分辨率对-cxr-分类的影响)
8. [CXR 分类最佳实践总结 (Best Practices)](#8-cxr-分类最佳实践总结)

---

## 1. RSNA Pneumonia Detection Challenge (2018)

### 竞赛概况

| 项目 | 详情 |
|------|------|
| 平台 | Kaggle |
| 参赛队伍 | **1,400+ teams**，346 teams 提交了最终结果 |
| 任务 | 在 CXR 上检测并定位肺炎区域 (object detection) |
| 评估指标 | mAP (Mean Average Precision) at IoU thresholds |
| 数据来源 | RSNA / NIH Clinical Center |

### 第 1 名方案 (Team: Ian Pan et al.)

**架构选择：**
- **50 个 object detection 模型**的 ensemble，跨 3 种不同架构：
  - RetinaNet (ResNet101 / ResNet152 backbone)
  - Deformable R-FCN (ResNet101 backbone)
  - Relation Networks
- Classification 部分使用 InceptionResNetV2 和 Xception
- 将 Keras 模型改为接受 **grayscale (1-channel) 输入**

**图像分辨率：**
- Detection 模型：**384px**
- Classification 模型：多种分辨率 (256, 320, 384, 448, 512px)

**训练策略：**
- 使用 **10-fold cross-validation**
- Model selection 基于 **highest AUC + F1 score** per fold
- 同时训练 binary classifier 和 multiclass classifier
- 仅在 Stage 1 training data 上训练，未在 Stage 2 上 retrain

**Ensemble 策略：**
- 分类器用于排除低概率图像的 box predictions
- 使用 IoU ≥ 0.4 进行 bounding box 合并

**关键启示 (对 single model 的启发)：**
- Transfer learning from ImageNet 在 CXR 上表现良好
- 适配 grayscale 输入是关键步骤
- AUC + F1 作为 model selection criterion 比单独用 loss 更好

---

## 2. CheXpert Competition (Stanford)

### 竞赛概况

| 项目 | 详情 |
|------|------|
| 平台 | Stanford ML Group 官方排行榜 |
| 提交数量 | **208+ ranked submissions**，150+ teams |
| 数据集 | **224,316 张** CXR，65,240 patients |
| 任务 | 5 个病理的 multi-label classification |
| 评估指标 | **Average AUC** (5 个病理) |
| 五个竞赛任务 | Atelectasis, Cardiomegaly, Consolidation, Edema, **Pleural Effusion** |

### Pleural Effusion 特别说明

- Baseline 模型在 **Pleural Effusion 上取得了最高的 AUC (0.97)**
- AUPRC 也最高 (0.91)
- 对于 Pleural Effusion，不同 uncertainty label 处理策略的差异**不具有统计显著性**
- 说明 Pleural Effusion 在 CXR 上相对容易检测（特征明显：液体积聚在肋膈角）

### 第 1 名方案: Deep AUC Maximization (DAM)

**核心方法：AUC Margin Loss + PESG Optimizer**

该方案由 LibAUC 团队提出，直接优化 AUC 而非传统的 cross-entropy loss：

| 组件 | 详情 |
|------|------|
| Loss Function | **AUC Margin Loss** (min-max surrogate) |
| Optimizer | **PESG** (Proximal Epoch Stochastic Gradient) |
| Architecture | DenseNet121 (也测试了 DenseNet161/169/201, ResNet, EfficientNet) |
| Batch Size | 32 |
| Margin Parameter (m) | 从 {0.1, 0.3, 0.5, 0.7, 1.0} 中调优 |
| 正则化 (γ) | 从 {1/300, 1/500, 1/800} 中调优 |
| 成绩 | **Mean AUC = 0.9305**，排名 1/150+ |

**训练流程：**
1. **Phase 1 (Pre-training)**: 使用 cross-entropy loss 预训练
2. **Phase 2 (AUC Maximization)**: 切换到 AUC Margin Loss + PESG optimizer
3. Learning rate 初始 0.1，在 50% 和 75% epoch 衰减
4. Weight decay: 1e-4

**为什么 AUC 优化对 CXR 重要：**
- CXR 数据集天然 class imbalance（正样本稀少）
- AUC 优化直接最大化"任意正样本分数高于负样本"的概率
- 比 cross-entropy 对 imbalanced data 更鲁棒

**Batch Score Normalization (BSN):**
- 对 mini-batch 内的预测分数做 L2 normalization
- 跨多个数据集一致提升性能

### 其他顶级方案关键发现

**架构对比 (CheXtransfer 研究)：**
- 传统架构 (DenseNet, ResNet) 在 CheXpert 上**优于** NAS 生成的架构 (EfficientNet, MobileNet)
- **ResNet152, DenseNet161, ResNet50** 在 CheXpert AUC 上表现最佳
- ImageNet pretrained weights 显著优于 random initialization

**Uncertainty Label 处理策略：**

| 策略 | 描述 | 最佳适用 |
|------|------|----------|
| U-Ones | 将 uncertain label 映射为 1 (positive) | Atelectasis, Edema |
| U-Zeros | 将 uncertain label 映射为 0 (negative) | - |
| U-MultiClass | 将 uncertain 作为独立类别 | Cardiomegaly |
| U-SelfTrained | 用模型预测重新标注 uncertain | 通用 |

> **对 Pleural Effusion**: 所有策略差异不显著，使用 U-Ones 即可。

---

## 3. SIIM-ACR Pneumothorax Segmentation (2019)

### 竞赛概况

| 项目 | 详情 |
|------|------|
| 平台 | Kaggle |
| 参赛队伍 | **1,475 teams**，352 teams 提交最终结果 |
| 任务 | 在 CXR 上分割气胸区域 |
| 评估指标 | Dice coefficient |

### 第 1 名方案

**架构：**
- AlbuNet (U-Net + **ResNet34** encoder)
- SCSEUnet (U-Net + **SE-ResNet50** encoder, Squeeze-and-Excitation)
- U-Net + **ResNet50** encoder

**图像分辨率：**
- 初始阶段：**512×512**
- 后期阶段：**1024×1024** (progressive upsampling)

**Loss Function (ComboLoss)：**

```
Loss = w1 * BCE + w2 * Dice + w3 * Focal
```

不同模型使用不同权重：
| 模型 | BCE:Dice:Focal |
|------|---------------|
| AlbuNet (validation) | 3:1:4 |
| AlbuNet (public) | 1:1:1 |
| ResNet50 | 2:1:2 |

**Data Augmentation (Albumentations)：**
- HorizontalFlip
- RandomBrightnessContrast
- 随机 gamma 调整
- ElasticTransform, GridDistortion, OpticalDistortion
- ShiftScaleRotate

**四阶段训练策略 (非常值得借鉴)：**

| 阶段 | Epochs | Learning Rate | Sample Rate | Scheduler |
|------|--------|--------------|-------------|-----------|
| Part 0 | 10-12 | 1e-3 → 1e-4 | 0.8 | ReduceLROnPlateau |
| Part 1 | - | ~1e-5 | 0.6 | CosineAnnealingLR |
| Part 2 | - | ~1e-5 | 0.4 | CosineAnnealingLR |
| Fine-tune | - | 1e-5 → 1e-6 | 0.5 | CosineAnnealingLR |

**Sliding Sample Rate 策略：**
- 将气胸/非气胸样本比例从 0.8 逐步降至 0.4
- 早期快速学习，后期适应真实数据分布
- **这对 class imbalance 问题非常有效！**

**其他关键技巧：**
- Checkpoint averaging: 每个 fold 选 top 3 个 checkpoint 取平均
- Test-Time Augmentation (TTA): 水平翻转
- Encoder freezing: 在分辨率上升阶段冻结 encoder
- 小 batch size (2-4 images) 在单张 1080Ti 上训练

**第 4 名方案补充：**
- 使用 ResNet34 backbone + frozen BatchNorm
- 带有 deep supervision branch 的 U-Net
- 空样本比例从 0.8 线性降至 0.22

---

## 4. VinBigData Chest X-ray Abnormalities Detection (2021)

### 竞赛概况

| 项目 | 详情 |
|------|------|
| 平台 | Kaggle |
| 参赛队伍 | **1,277 teams**，来自 60+ 个国家 |
| 奖金 | $50,000 |
| 数据集 | 18,000 张 PA 位 CXR，17 位放射科医生标注 |
| 任务 | 14 种异常的定位与分类 |
| 评估指标 | mAP at IoU 0.4 |
| 竞赛时间 | 2020.12.31 - 2021.03.31 |

### 第 1 名方案 (Team ℳS²Ƒ)

**核心技术：**
- Detection 模型为主 (DetectoRS, YOLOv5, Cascade R-CNN, RetinaNet)
- 多模型 ensemble
- Label smoothing
- Focal Loss for classification head

### 第 2 名方案 (ZFTurbo)

- 三个团队成员各自训练模型后进行 ensemble
- 包含 detection + classification 的组合流水线

### 关键技术总结

- **高分辨率输入**对于小病灶检测至关重要
- **Label smoothing** 有助于处理放射科医生之间的标注不一致
- **Multi-scale detection** 策略适合不同尺度的异常

---

## 5. RANZCR CLiP Catheter and Line Position Challenge (2021)

### 竞赛概况

| 项目 | 详情 |
|------|------|
| 平台 | Kaggle |
| 参赛队伍 | **1,549 teams** |
| 数据集 | 30,083 张高分辨率 CXR，50,612 image-level labels |
| 任务 | 11 个类别的 multi-label classification (导管/管路位置) |
| 评估指标 | AUC |

### Top 方案特征

**第 7 名方案：**
- **NFNet-F1** 多阶段训练
- Segmentation 辅助任务
- External data + pseudo-labeling

**Top-5% 方案 (71st / 1,549)：**
- 5+2 CNN 模型 ensemble (PyTorch)
- Test AUC = **0.971**
- BCEWithLogitsLoss + Focal Loss

**通用策略：**
- 大模型效果更好：ResNet200D, EfficientNet-B7, NFNet-F1
- 图像分辨率 > 1024px (最高 1536px)
- Teacher-student training + pseudo-labeling
- External data augmentation

---

## 6. CheXNet / NIH ChestX-ray14 基准研究

### 原始 CheXNet

| 组件 | 配置 |
|------|------|
| Architecture | **DenseNet-121** (ImageNet pretrained) |
| Input Size | **224×224** |
| Optimizer | Adam, lr=0.001 |
| Batch Size | 16 |
| Loss | Binary Cross-Entropy |
| Augmentation | Random horizontal flip, resize |
| Dataset | 112,120 frontal view X-rays, 30,805 patients |

### 改进版 CheXNet (DannyNet)

仅 **三个改动** 就显著提升了性能：

| 改进 | 原始 → 改进 | 影响 |
|------|-------------|------|
| Loss Function | BCE → **Focal Loss** (γ=2, α=1) | 显著降低 test loss，提高少数类预测信心 |
| Optimizer | Adam → **AdamW** (lr=5e-5, weight decay) | 更好的泛化 |
| Augmentation | Flip only → + **ColorJitter** | 增强鲁棒性 |

**性能对比：**

| 指标 | CheXNet Replica | DannyNet (改进版) |
|------|-----------------|-------------------|
| Test AUC | 0.79 | **0.85** |
| Average F1 | 0.08 | **0.39** |
| Test Loss | 0.17 | **0.04** |

**Scheduler**: ReduceLROnPlateau

> **关键发现：Focal Loss + AdamW + ColorJitter 这三个简单改动就将 AUC 从 0.79 提升到 0.85，F1 从 0.08 提升到 0.39。**

---

## 7. 图像分辨率对 CXR 分类的影响

基于 MIMIC-CXR-JPG 数据集的系统研究 (原始分辨率 ~2500×3056)：

### 不同分辨率下各病理的最优表现

| 病理类型 | 最优分辨率 | 原因 |
|----------|-----------|------|
| Cardiomegaly, Enlarged Cardiomediastinum | **256×256 ~ 512×512** | 大结构，需要大 receptive field |
| Pneumonia, Edema | **512×512 ~ 1024×1024** | 中等尺度特征 |
| Pneumothorax, Fracture, Lung Lesion | **1024×1024 ~ 2048×2048** | 细小特征，需要高分辨率 |
| **Pleural Effusion** | **256×256 ~ 512×512** | 大面积液体积聚，低分辨率即可识别 |

### 关键结论

1. **没有单一最优分辨率** — 不同病理需要不同分辨率
2. 更高分辨率**不一定**更好 — Effective Receptive Field (ERF) 会随分辨率增大而缩小
3. **Multi-resolution ensemble** 比任何单一分辨率平均高 4.3%
4. 对于 **Pleural Effusion** 这种大面积特征，**224×224 到 512×512 已经足够**
5. 超过 256×256 后，Pleural Effusion 的性能提升趋于平缓

> **对本项目的启示：ResNet50 使用 224×224 输入即可有效检测 Pleural Effusion，无需高分辨率。**

---

## 8. CXR 分类最佳实践总结

### 8.1 Architecture 选择

| 推荐 | 理由 |
|------|------|
| **ResNet50** (ImageNet pretrained) | CheXtransfer 研究表明 ResNet50 在 CheXpert 上表现优秀，优于 EfficientNet |
| DenseNet121 作为备选 | CheXNet 基准架构，参数更少 |
| 避免过大模型 | 对于 binary classification，ResNet50 已足够 |

**Grayscale 输入适配：**
- 方法 1：将单通道复制为 3 通道以兼容 ImageNet pretrained weights
- 方法 2：修改第一个 conv 层为 1-channel 输入（需重新初始化该层权重）
- **推荐方法 1**，保留更多 pretrained 信息

### 8.2 Loss Function

**强烈推荐 Focal Loss 替代 BCE：**

```python
# Focal Loss 配置
gamma = 2.0  # focusing parameter，增大对 hard examples 的关注
alpha = 1.0  # 或根据 class ratio 设置 (如 0.75 for positive class)
```

| Loss | 适用场景 | 效果 |
|------|----------|------|
| **Focal Loss** (γ=2, α=1) | 有 class imbalance | 显著降低 loss，提高 minority class 预测信心 |
| Weighted BCE | 简单 class imbalance | 基础有效 |
| AUC Margin Loss (LibAUC) | 追求极致 AUC | CheXpert 第 1 名，但实现复杂 |
| BCE + Dice (ComboLoss) | 分割任务 | 不适用于纯分类 |

### 8.3 Optimizer & Learning Rate Schedule

**推荐配置：**

```python
# Option A: 简单有效 (推荐作为课程项目)
optimizer = AdamW(params, lr=5e-5, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# Option B: 更精细的调度
optimizer = AdamW(params, lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Option C: Warmup + Cosine Annealing (竞赛常用)
# 5 epochs warmup → cosine decay to 1e-6
```

**Learning Rate 推荐范围：**
- ImageNet pretrained ResNet50: **1e-4 ~ 5e-5**
- Fine-tuning 最后几层: **1e-3 ~ 1e-4**
- 全网络 fine-tuning: **5e-5 ~ 1e-5**

### 8.4 Data Augmentation

**CXR 安全的 Augmentation (推荐使用 Albumentations)：**

```python
import albumentations as A

train_transform = A.Compose([
    # === 基础增强 (必须使用) ===
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),                    # CXR 左右对称，安全

    # === 颜色增强 (CheXNet 改进中被证实有效) ===
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.ColorJitter(p=0.3),                        # DannyNet 的关键改进之一

    # === 几何增强 (小幅度) ===
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=10,                          # CXR 旋转要小！(5-15度)
        p=0.5
    ),

    # === 医学图像特有 ===
    A.CLAHE(clip_limit=2.0, p=0.3),              # 增强对比度，适合 CXR

    # === Normalize (ImageNet 标准化) ===
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

**CXR 需要避免的 Augmentation：**

| 不推荐 | 原因 |
|--------|------|
| VerticalFlip | CXR 不会上下翻转 |
| 大角度旋转 (>15°) | 真实 CXR 很少大角度倾斜 |
| CutOut / CutMix | 可能遮盖病灶关键区域 |
| 过强的 color augmentation | 灰度图像信息可能被破坏 |

### 8.5 Class Imbalance 处理

针对 **Pleural Effusion binary classification** 的推荐策略：

| 策略 | 优先级 | 说明 |
|------|--------|------|
| **Focal Loss** | ★★★★★ | 最简单有效，γ=2 自动降低 easy sample 权重 |
| **Weighted Sampling** | ★★★★ | 对 minority class 过采样或设置 sampler |
| **Class-weighted Loss** | ★★★★ | `pos_weight` 按正负样本比例设置 |
| Sliding Sample Rate | ★★★ | 训练过程中动态调整正负样本比例 |
| Oversampling | ★★★ | 简单但可能过拟合 |
| Label Smoothing | ★★★ | 处理标注噪声，设置 smoothing=0.1 |
| Synthetic Data (GAN) | ★★ | 实现复杂，收益不确定 |

**Focal Loss + Weighted Sampling 组合推荐：**

```python
# Class-weighted BCE (如果正样本占比 20%)
pos_weight = torch.tensor([4.0])  # negative/positive ratio
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 或 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

### 8.6 Fine-tuning 策略

**分阶段 Fine-tuning (推荐)：**

| 阶段 | 操作 | Epochs | LR |
|------|------|--------|----|
| Stage 1 | 冻结 ResNet50 backbone，只训练 classifier head | 3-5 | 1e-3 |
| Stage 2 | 解冻最后 2 个 ResNet block (layer3, layer4) | 5-10 | 1e-4 |
| Stage 3 | 解冻全部层 | 5-10 | 1e-5 ~ 5e-5 |

**Batch Normalization 处理：**
- Fine-tuning 初期建议 **freeze BN layers** (使用 ImageNet 统计量)
- CXR 与 ImageNet 差异大时，后期可 unfreeze BN
- SIIM-ACR 第 4 名方案使用了 frozen BatchNorm

### 8.7 Test-Time Augmentation (TTA)

**推荐的 TTA 策略 (无额外成本提升 AUC)：**

```python
# 最简单有效的 TTA：水平翻转
def tta_predict(model, image):
    pred1 = model(image)
    pred2 = model(torch.flip(image, dims=[-1]))  # horizontal flip
    return (pred1 + pred2) / 2
```

- HorizontalFlip TTA 平均可提升 **0.25% AUC**
- HorizontalFlip + Zoom TTA 可提升 **0.51% AUC**
- 几乎零成本（仅增加一次 forward pass）

### 8.8 Model Selection & Evaluation

**推荐 Model Selection 策略：**
- 使用 **validation AUC** (不是 validation loss) 作为 model selection 指标
- RSNA 第 1 名使用 AUC + F1 组合
- 保存 top-K checkpoints 并做 checkpoint averaging
- 使用 **patient-level split** (避免同一患者的不同图像出现在 train 和 val 中)

**Per-class Threshold 优化：**
- 默认 threshold 0.5 通常不是最优
- 在 validation set 上搜索最优 threshold（最大化 F1 或 Youden's J）

### 8.9 针对 Pleural Effusion 的特殊考虑

基于以上所有竞赛的综合分析：

1. **Pleural Effusion 是 CXR 中相对容易检测的病理**
   - CheXpert baseline AUC = 0.97
   - 特征明显：液体在重力作用下积聚在肋膈角

2. **推荐分辨率：224×224**
   - Pleural Effusion 是大面积特征
   - 研究表明 256×256 甚至更低分辨率即可有效检测
   - 不需要高分辨率来捕获细节

3. **关键特征区域：肋膈角 (costophrenic angle)**
   - 模型的 attention 应集中在肺底部
   - 可使用 GradCAM 验证模型是否关注正确区域

4. **CLAHE 预处理可能有帮助**
   - Histogram stretching 在 Pleural Effusion 检测中表现最佳
   - CLAHE 增强局部对比度，有助于区分液体和肺组织

---

## 附录：各竞赛规模对比

| 竞赛 | 年份 | Teams | 任务类型 | Top 方案核心 |
|------|------|-------|----------|-------------|
| RSNA Pneumonia Detection | 2018 | 1,400+ | Detection | RetinaNet ensemble |
| CheXpert | 2019-2020 | 150+ | Multi-label Classification | Deep AUC Maximization |
| SIIM-ACR Pneumothorax | 2019 | 1,475 | Segmentation | U-Net + ComboLoss |
| VinBigData CXR | 2021 | 1,277 | Detection | Multi-detector ensemble |
| RANZCR CLiP | 2021 | 1,549 | Multi-label Classification | NFNet/EfficientNet + pseudo-labeling |

---

## 附录：针对本项目的推荐 Pipeline

```
ResNet50 (ImageNet pretrained, 3-channel input)
    ↓
Input: 224×224 (grayscale → 3-channel copy)
    ↓
Stage 1: Freeze backbone, train head (3 epochs, lr=1e-3, Adam)
    ↓
Stage 2: Unfreeze all, fine-tune (15-20 epochs)
    - Optimizer: AdamW, lr=5e-5, weight_decay=1e-4
    - Loss: Focal Loss (γ=2, α=0.75) 或 Weighted BCE
    - Scheduler: CosineAnnealingLR (eta_min=1e-6) 或 ReduceLROnPlateau
    - Augmentation: HorizontalFlip + RandomBrightnessContrast + ShiftScaleRotate + CLAHE
    - Batch Size: 32-64
    ↓
Model Selection: Best validation AUC checkpoint
    ↓
Inference: + HorizontalFlip TTA
    ↓
Threshold: 在 validation set 上优化 threshold (maximize F1)
```

---

## 参考来源

- [RSNA Pneumonia Detection Challenge - Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
- [RSNA 1st Place Solution - GitHub](https://github.com/i-pan/kaggle-rsna18)
- [CheXpert Competition - Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [CheXpert Competition Models - GitHub](https://github.com/kamenbliznashki/chexpert)
- [SIIM-ACR Pneumothorax 1st Place Solution - GitHub](https://github.com/sneddy/pneumothorax-segmentation)
- [VinBigData 1st Place Solution - Kaggle](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/writeups/s-1st-place-solution)
- [VinBigData 2nd Place Solution - GitHub](https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection)
- [RANZCR CLiP Challenge - Kaggle](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification)
- [Deep AUC Maximization (LibAUC) Paper](https://arxiv.org/abs/2012.03173)
- [LibAUC - Official Site](https://libauc.org/)
- [CheXtransfer Paper](https://arxiv.org/abs/2101.06871)
- [Reproducing and Improving CheXNet](https://arxiv.org/html/2505.06646v1)
- [Effect of Image Resolution on CXR Classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC10403240/)
- [Deep Learning for Pleural Effusion Detection](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-024-01260-1)
- [Chest X-ray Effusion Detection using CNN-ResNet - GitHub](https://github.com/GvHemanth/Chest-X-Ray-Effusion-Detection-using-CNN-ResNet)
- [RANZCR CLiP Top-5% Solution - GitHub](https://github.com/kozodoi/Kaggle_RANZCR_Challenge)
- [Tackling RSNA Pneumonia Detection Challenge - AJR](https://ajronline.org/doi/full/10.2214/AJR.19.21512)
