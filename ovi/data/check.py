import os
import sys
import stat
# === 这里填入你报错的那个具体路径 ===
TARGET_PATH = '/mnt/vision-gen-ks3/Video_Generation/DataSets/CustomDataSet/talk_data/openhumanvid/av_synced_cut_audio/be3f227dec20fa32d3204b642fcdbcc3.wav'
def check_path_diagnostic(path):
    print("="*60)
    print("🕵️‍♂️ 路径深度诊断工具 (Path Diagnostic Tool)")
    print("="*60)
    # 1. 基础字符串检查
    print(f"\n[1] 字符串检查:")
    print(f"    目标路径: {path}")
    print(f"    Repr显示: {repr(path)}")
    if '\n' in path or '\r' in path:
        print("    ❌ 警告: 发现换行符！")
    else:
        print("    ✅ 字符串格式正常 (无换行符)")
    # 2. 当前用户环境
    print(f"\n[2] 环境检查:")
    try:
        import pwd
        user_name = pwd.getpwuid(os.getuid()).pw_name
    except:
        user_name = str(os.getuid())
    print(f"    当前用户: {user_name} (UID: {os.getuid()}, GID: {os.getgid()})")
    print(f"    工作目录: {os.getcwd()}")
    # 3. 逐级路径探测 (这是最关键的一步)
    print(f"\n[3] 逐级路径存在性检查:")
    parts = path.split(os.sep)
    current_check = "/"
    stop_check = False
    # 从根目录开始一级一级往下走
    for i, part in enumerate(parts):
        if not part: continue # 跳过空字符
        current_check = os.path.join(current_check, part)
        # 检查存在性
        exists = os.path.exists(current_check)
        is_link = os.path.islink(current_check)
        status = "✅ 存在" if exists else "❌ 不存在"
        if is_link:
            status += " (🔗 软链接)"
            try:
                real_path = os.path.realpath(current_check)
                status += f" -> 指向: {real_path}"
                if not os.path.exists(real_path):
                    status += " [❌ 断链!]"
            except:
                status += " [无法解析链接]"
        # 检查权限
        perm_str = ""
        if exists:
            try:
                st = os.stat(current_check)
                perm_str = f" [权限: {stat.filemode(st.st_mode)}, Owner: {st.st_uid}]"
                if not os.access(current_check, os.R_OK):
                    perm_str += " 🚫 无读取权限!"
                if not os.access(current_check, os.X_OK) and os.path.isdir(current_check):
                    perm_str += " 🚫 无进入权限!"
            except Exception as e:
                perm_str = f" [无法获取状态: {e}]"
        print(f"    Level {i}: {current_check:<60} {status}{perm_str}")
        if not exists:
            print(f"\n    🚨 诊断结果: 路径在 '{current_check}' 处中断！")
            # 如果断点是目录，尝试列出该级父目录的内容，看看有没有拼写错误
            parent_dir = os.path.dirname(current_check)
            if os.path.exists(parent_dir):
                try:
                    siblings = os.listdir(parent_dir)
                    print(f"    ℹ️  父目录 '{parent_dir}' 下有 {len(siblings)} 个项目。")
                    print(f"       前10个: {siblings[:10]}")
                    if part in siblings:
                        print(f"       🤔 奇怪: '{part}' 在listdir中显示存在，但exists返回False。这通常是挂载点失效或断链。")
                    else:
                        # 模糊匹配检查
                        import difflib
                        matches = difflib.get_close_matches(part, siblings)
                        if matches:
                            print(f"       💡 你是不是想找: {matches}")
                except PermissionError:
                    print(f"       🚫 无法读取父目录内容 (Permission Denied)")
            stop_check = True
            break
    if not stop_check:
        print(f"\n    🎉 恭喜: 文件检查完全通过！Python 应该能读取它。")
        # 尝试真的读取一下
        try:
            size = os.path.getsize(path)
            print(f"    📄 文件大小: {size / 1024:.2f} KB")
            with open(path, 'rb') as f:
                f.read(10)
            print("    📖 尝试打开读取: 成功")
        except Exception as e:
            print(f"    ☠️  但是打开文件失败: {e}")
if __name__ == "__main__":
    check_path_diagnostic(TARGET_PATH)
