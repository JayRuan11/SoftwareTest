import os
import subprocess

# ================= 配置区域 =================

# 输出目录
OUTPUT_DIR = "./output_mismatched_videos"

# 视频路径定义
VIDEO_1 = "/mnt/vision-gen-ks3/Video_Generation/DataSets/CustomDataSet/talk_data/openhumanvid/av_synced_cut_clips/4a6489739ba52adaaaf00f13264d35de_3.mp4"
# VIDEO_2 = "/mnt/vision-gen-ks3/Video_Generation/DataSets/CustomDataSet/talk_data/openhumanvid/av_synced_cut/000ef40f09d6246078d7265580684a8e.mp4"
# VIDEO_3 = "/mnt/vision-gen-ks3/Video_Generation/DataSets/CustomDataSet/talk_data/openhumanvid/av_synced_cut/001bce0c028dedb71c26ad943b74f4e7.mp4"

# 任务列表：每个任务包含一个视频源和一组需要替换的MP3路径
tasks = [
    # 任务组 1: 金发女性 (对应文件夹 test/6)
    {
        "video_source": VIDEO_1,
        "mp3_list": [
            "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/mismatched_3/4a6489739ba52adaaaf00f13264d35de_3_mismatch_adults_adults_m2.wav",
            "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/mismatched_3/4a6489739ba52adaaaf00f13264d35de_3_mismatch_children_children_f3.wav",
            "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/mismatched_3/4a6489739ba52adaaaf00f13264d35de_3_mismatch_elderly_elderly_f4.wav",
            "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/mismatched_3/4a6489739ba52adaaaf00f13264d35de_3_mismatch_teens_teens_m4.wav",
        ]
    },
    # 任务组 2: 黑人 Morpheus (对应文件夹 test/1)
    # {
    #     "video_source": VIDEO_2,
    #     "mp3_list": [
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/1/clone_f_1_1.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/1/clone_f_4_5.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/1/clone_f_7_4.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/1/clone_m_1_1.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/1/clone_m_5_1.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/1/clone_m_8_3.mp3",
    #     ]
    # },
    # # 任务组 3: 中年男子 (对应文件夹 test/2)
    # {
    #     "video_source": VIDEO_3,
    #     "mp3_list": [
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/2/clone_f_1_1.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/2/clone_f_4_5.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/2/clone_f_7_5.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/2/clone_m_1_1.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/2/clone_m_4_5.mp3",
    #         "/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/data/test/2/clone_m_8_5.mp3",
    #     ]
    # }
]

# ================= 脚本逻辑 =================

def process_videos():
    # 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_count = 0

    for task_idx, task in enumerate(tasks):
        video_src = task["video_source"]
        mp3_files = task["mp3_list"]
        
        # 简单的视频名标识 (用于日志)
        video_label = f"Video_{task_idx+1}"

        if not os.path.exists(video_src):
            print(f"Error: Video source not found: {video_src}")
            continue

        print(f"--- Processing Group {task_idx+1} ({len(mp3_files)} audios) ---")

        for mp3_path in mp3_files:
            mp3_path = mp3_path.strip()
            if not os.path.exists(mp3_path):
                print(f"  [Skip] Audio not found: {mp3_path}")
                continue

            # 生成输出文件名
            # 为了防止文件名重复 (不同组都有 clone_f_1_1.mp3)，文件名加入父文件夹名称
            # 格式: 视频名_音频父目录_音频名.mp4
            vid_basename = os.path.splitext(os.path.basename(video_src))[0]
            audio_basename = os.path.splitext(os.path.basename(mp3_path))[0]
            audio_folder = os.path.basename(os.path.dirname(mp3_path)) # 例如 '6', '1', '2'

            output_filename = f"{vid_basename}_test{audio_folder}_{audio_basename}.mp4"
            output_full_path = os.path.join(OUTPUT_DIR, output_filename)

            # FFmpeg 命令
            cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-i", video_src,      # 输入视频
                "-i", mp3_path,       # 输入音频
                "-c:v", "copy",       # 视频流复制 (无损且快)
                "-c:a", "aac",        # 音频编码 AAC
                "-map", "0:v:0",      # 取第一个文件的视频流
                "-map", "1:a:0",      # 取第二个文件的音频流
                "-y",                 # 覆盖输出
                output_full_path
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"  [OK] {output_filename}")
                total_count += 1
            except subprocess.CalledProcessError as e:
                print(f"  [Fail] Could not create {output_filename}. Error: {e}")

    print(f"\nCompleted. Generated {total_count} videos in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    process_videos()