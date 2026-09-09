import os
import torch
import sys
from ovi.data.loader import AspectRatioDataLoader
# ================= 配置区域 (请修改这里) =================
# 1. 你的真实 JSON 文件绝对路径
JSON_PATH = "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/ovi_test/dataset"
# 2. 你的数据集中期望加载的帧数
#    如果你的视频长度不一，建议保持 frame_tolerance 为较大的值
NUM_FRAMES = 25
# 3. 你的 JSON 中描述字段的 Key
#    (例如: "qwen2-VL-72B-detail" 或 "Qwen3Omni" 等)
DESCRIPTION_KEY = "Qwen3Omni"
# ========================================================
def test_real_dataloader():
    print(f"正在读取真实数据: {JSON_PATH}")
    # 检查文件是否存在
    try:
        # 初始化 DataLoader
        # 注意: frame_tolerance 设置得很大 (1000)，以防你的真实视频长度差异大导致被过滤
        loader = AspectRatioDataLoader(
            data_paths=[JSON_PATH],
            batch_size=2,
            num_workers=0,  # 调试模式建议单线程
            num_aspect_buckets=1,
            description_model=DESCRIPTION_KEY,
            num_frames=NUM_FRAMES,
            frame_tolerance=1000,
            height=256,
            width=256
        )
        dataset_len = len(loader.dataset)
        print(f"✅ 成功加载 Dataset. 总样本数: {dataset_len}")
        if dataset_len == 0:
            print("❌ 警告: Dataset 长度为 0。可能是所有数据都被过滤了 (检查路径是否正确，或者帧数是否符合要求)。")
            return
        print(f"Bucket keys: {loader.dataset.get_bucket_names()}")
        print("\n开始迭代前 3 个 Batch 进行检查...")
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 3: # 只测试前 3 个 batch，避免刷屏
                break
            print(f"\n--- Batch {batch_idx} ---")
            # --- 检查 1: 关键 Key ---
            # 根据你的模型需求，这里列出必须存在的 Key
            required_keys = ['video', 'audio', 'text', 'mismatched_audio'] # 'mismatched_audio' 如果没有负样本可以去掉
            keys_exist = all(k in batch for k in required_keys)
            if keys_exist:
                print(f"✅ 关键 Key 存在: {list(batch.keys())}")
            else:
                print(f"❌ 缺少 Key. 当前 Keys: {list(batch.keys())}")
            # --- 检查 2: Tensor 形状 ---
            if 'video' in batch:
                video = batch['video']
                print(f"Video Shape: {video.shape} (Expect: [B, C, {NUM_FRAMES}, H, W])")
                # 验证帧数
                if video.shape[2] != NUM_FRAMES:
                    print(f"⚠️  警告: 加载的帧数 ({video.shape[2]}) 与设定 ({NUM_FRAMES}) 不一致 (可能是 padding 或 resize 导致)")
            if 'audio' in batch:
                print(f"Audio Shape: {batch['audio'].shape}")
            if 'text' in batch:
                print(f"Text Sample [0]: {batch['text'][0]}")
            if 'mismatched_audio' in batch:
                print(f"mismatched_audio Shape: {batch['mismatched_audio'].shape}")
            # --- 检查 3: 路径是否存在 (如果你想检查原始路径) ---
            # 如果 batch 包含 path 信息 (取决于你的 Dataset 是否返回 path)
            if 'video_path' in batch:
                print(f"Video Path [0]: {batch['video_path'][0]}")
        print("\n🎉 测试结束。")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    test_real_dataloader()
