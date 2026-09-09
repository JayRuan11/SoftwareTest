# 混合训练（Multimodal + Audio-Only）使用指南

## 概述

混合训练模式允许同时使用多模态数据（视频+音频）和纯音频数据进行训练，特别适用于：
- 中文音频效果较差，但缺少对应的多模态视频语料
- 想利用大量纯音频数据提升音频生成质量
- 同时保持视频-音频对齐能力

## 核心设计

### 1. 数据格式

#### 多模态数据（Multimodal）
metadata.json 格式保持不变：
```json
{
  "video_path": "/path/to/video.mp4",
  "audio_path": "/path/to/audio.wav",
  "description": {
    "qwen2-VL-72B-detail": "视频描述文本"
  },
  "quality": {
    "width": 720,
    "height": 720,
    "frameNum": 81
  }
}
```

#### 纯音频数据（Audio-Only）
添加 `audio_only: true` 标记或省略 `video_path`：
```json
{
  "audio_only": true,
  "audio_path": "/path/to/audio.wav",
  "description": {
    "qwen2-VL-72B-detail": "音频描述文本"
  }
}
```

### 2. 模型前向传播

利用 FusionModel 已有的单模态支持：
- **多模态数据**：正常的 fusion 模式，video + audio 交叉注意力
- **纯音频数据**：传入 `vid=None`，模型自动切换到 audio-only 模式
  - 跳过 video model 的计算
  - 跳过 fusion cross-attention
  - 只运行 audio model

### 3. 损失计算策略

#### 多模态损失
```
loss = video_loss * video_loss_weight + audio_loss * (1 - video_loss_weight)
     = video_loss * 0.85 + audio_loss * 0.15
```

#### 纯音频损失
```
loss = audio_loss * audio_only_loss_weight
```

**关键参数 `audio_only_loss_weight`**：
- 默认：1.0
- 推荐：1.5 - 3.0
- 原因：纯音频 latent 维度较小，损失量级可能比多模态小，需要放大以平衡梯度

### 4. 训练参数配置

#### 推荐的可训练参数组合

**场景 A：主要提升音频质量（推荐）**
```yaml
train_fusion: true   # 保持 fusion 能力
train_audio: true    # 重点训练音频
train_video: false   # 冻结视频模型，节省计算
```

**场景 B：全面微调**
```yaml
train_fusion: true
train_audio: true
train_video: true    # 同时微调视频，需要更多显存
```

**场景 C：只训练 fusion 层**
```yaml
train_fusion: true
train_audio: false
train_video: false   # 最省显存
```

## 使用步骤

### 步骤 1：准备数据

创建混合数据集的 metadata.json：
```json
[
  {
    "video_path": "/data/multimodal/video1.mp4",
    "audio_path": "/data/multimodal/audio1.wav",
    "description": {"qwen2-VL-72B-detail": "一段音乐视频"}
  },
  {
    "audio_only": true,
    "audio_path": "/data/audio_only/speech1.wav",
    "description": {"qwen2-VL-72B-detail": "一段中文语音"}
  },
  {
    "audio_only": true,
    "audio_path": "/data/audio_only/music1.wav",
    "description": {"qwen2-VL-72B-detail": "钢琴独奏"}
  }
]
```

### 步骤 2：配置训练参数

复制并修改配置文件：
```bash
cp ovi/configs/train/mixed_training_example.yaml ovi/configs/train/my_mixed_training.yaml
```

关键配置项：
```yaml
enable_mixed_training: true          # 必须启用
audio_only_loss_weight: 2.0          # 根据实际损失量级调整
train_fusion: true
train_audio: true
train_video: false
```

### 步骤 3：启动训练

```bash
python train_lightning.py \
    --config ovi/configs/train/my_mixed_training.yaml \
    --data_paths /path/to/dataset1 /path/to/dataset2 \
    --num_nodes 1
```

### 步骤 4：监控训练

观察 TensorBoard 日志中的关键指标：

**多模态样本：**
- `train/multimodal_loss`: 总损失
- `train/video_loss`: 视频损失
- `train/audio_loss`: 音频损失

**纯音频样本：**
- `train/audio_only_loss`: 音频损失（原始值）
- `train/loss`: 加权后的损失（audio_only_loss * audio_only_loss_weight）

**理想情况：**
- `train/multimodal_loss` 和 `train/loss`（audio-only）应该在相近的量级
- 如果相差超过 5-10 倍，调整 `audio_only_loss_weight`

## 损失量级调整指南

### 问题诊断

查看训练日志中两种样本的损失：
```
# 多模态样本
train/multimodal_loss: 0.0450
train/video_loss: 0.0500
train/audio_loss: 0.0150

# 纯音频样本
train/audio_only_loss: 0.0080
train/loss: 0.0160  (audio_only_loss * 2.0)
```

### 调整策略

**如果 audio_only loss 远小于 multimodal loss：**
```yaml
# 增大 audio_only_loss_weight
audio_only_loss_weight: 3.0  # 从 2.0 增加到 3.0
```

**如果 audio_only loss 远大于 multimodal loss：**
```yaml
# 减小 audio_only_loss_weight
audio_only_loss_weight: 1.0  # 从 2.0 减少到 1.0
```

**目标：** 两种样本的加权损失在相近量级（差异 < 2-3 倍）

## 数据混合比例建议

根据目标选择数据混合比例：

**保守策略（推荐起步）：**
- 70% 多模态数据
- 30% 纯音频数据
- 目标：保持视频-音频对齐能力，适度提升音频质量

**激进策略（有充足多模态数据）：**
- 50% 多模态数据
- 50% 纯音频数据
- 目标：显著提升音频质量

**音频优先策略（谨慎使用）：**
- 30% 多模态数据
- 70% 纯音频数据
- 风险：可能损害视频-音频对齐能力
- 建议：定期在验证集上测试对齐质量

## 潜在问题与解决方案

### 问题 1：Fusion layers 训练不足

**现象：** 多模态样本的视频-音频对齐变差

**解决：**
1. 增加多模态数据比例
2. 确保 `train_fusion: true`
3. 降低 `audio_only_loss_weight`，避免纯音频梯度主导

### 问题 2：纯音频效果没有改善

**原因分析：**
1. `audio_only_loss_weight` 太小，梯度被多模态主导
2. `train_audio: false`，音频模型参数被冻结
3. 纯音频数据质量或描述文本质量不高

**解决：**
1. 增大 `audio_only_loss_weight` (2.0 → 3.0 → 5.0)
2. 确保 `train_audio: true`
3. 检查纯音频数据质量和文本描述准确性

### 问题 3：显存不足

**解决方案：**
1. 减小 `batch_size`
2. 增加 `accumulate_grad_batches`
3. 设置 `train_video: false` 冻结视频模型
4. 使用 `gradient_checkpoint: true`
5. 使用 DeepSpeed Stage 2/3:
   ```yaml
   training_strategy: "deepspeed_stage_2"
   ```

### 问题 4：训练速度变慢

**原因：** 纯音频数据仍需要文本编码和音频 VAE 编码

**优化：**
1. 增加 `dataloader_num_workers`
2. 启用 `text_encoder_cpu_offload: true`
3. 纯音频数据跳过视频 VAE，已节省大量计算

## 验证建议

定期在验证集上测试：

**多模态能力：**
- 生成视频+音频，检查对齐质量
- 特别关注音画同步、情绪一致性

**纯音频能力：**
- 用纯音频提示词生成，评估音频质量
- 特别关注中文语音清晰度、音乐质量

**推荐验证频率：**
- 每个 epoch 结束后
- 每 N 个 checkpoint（如 save_top_k=3）

## 高级技巧

### 渐进式训练策略

**阶段 1：热身（10-20% epoch）**
```yaml
enable_mixed_training: false  # 仅多模态数据
train_fusion: true
train_audio: true
```

**阶段 2：混合训练（70-80% epoch）**
```yaml
enable_mixed_training: true   # 引入纯音频
audio_only_loss_weight: 2.0
```

**阶段 3：精调（最后 10% epoch）**
```yaml
audio_only_loss_weight: 1.5   # 降低权重，平衡两种模态
```

### 动态损失权重（高级）

可以在代码中实现动态调整：
```python
# 在 training_step 中
current_epoch_ratio = self.current_epoch / self.trainer.max_epochs
if current_epoch_ratio < 0.2:
    # 早期：只用多模态
    audio_only_weight = 0.0
elif current_epoch_ratio < 0.8:
    # 中期：正常混合
    audio_only_weight = self.audio_only_loss_weight
else:
    # 后期：降低权重
    audio_only_weight = self.audio_only_loss_weight * 0.5
```

## 总结

混合训练是一个强大的策略，但需要仔细调参：

✅ **关键成功因素：**
1. 合理的数据混合比例（建议 70%:30% 起步）
2. 正确的损失权重设置（audio_only_loss_weight）
3. 适当的可训练参数选择（train_fusion/audio/video）
4. 持续的验证和监控

⚠️ **注意事项：**
1. 监控两种样本的损失量级，及时调整权重
2. 定期验证多模态对齐能力，避免退化
3. 纯音频数据的文本描述质量很重要
4. 从保守策略开始，逐步增加纯音频比例

🎯 **预期效果：**
- 中文音频质量显著提升
- 保持视频-音频对齐能力
- 训练稳定，无灾难性遗忘
