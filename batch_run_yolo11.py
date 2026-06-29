import os
import sys
import time
import subprocess
from pathlib import Path


# =========================
# 配置区
# =========================

PROJECT_ROOT = Path(r"D:\PanopticSaliency")

# 强制使用项目虚拟环境里的 Python
PYTHON_EXE = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"

# 你的单张运行脚本
SCRIPT_PATH = PROJECT_ROOT / "demo.py"

# 数据集图片目录
IMG_ROOT = PROJECT_ROOT / "data" / "F-360iSOD-test" / "stimulis"

# 输出目录，注意 demo.py 里 OUT_DIR 也要改成 outputs_yolo11
OUT_DIR = PROJECT_ROOT / "outputs_yolo11"

# 先跑 5 张测试，没问题后改成 None
MAX_IMAGES = None

IMG_SUFFIX = [".png", ".jpg", ".jpeg", ".bmp"]

CONTINUE_ON_ERROR = True


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_image_list(img_root: Path):
    img_list = []
    for p in img_root.iterdir():
        if p.suffix.lower() in IMG_SUFFIX:
            img_list.append(p)
    return sorted(img_list, key=lambda x: x.name)


def main():
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"找不到虚拟环境 Python: {PYTHON_EXE}")

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"找不到主脚本: {SCRIPT_PATH}")

    if not IMG_ROOT.exists():
        raise FileNotFoundError(f"找不到图片目录: {IMG_ROOT}")

    ensure_dir(OUT_DIR)

    img_list = get_image_list(IMG_ROOT)

    if MAX_IMAGES is not None:
        img_list = img_list[:MAX_IMAGES]

    if len(img_list) == 0:
        raise RuntimeError(f"没有找到图片: {IMG_ROOT}")

    runtime_log_path = OUT_DIR / "batch_runtime_log.txt"
    error_log_path = OUT_DIR / "batch_error_log.txt"
    summary_path = OUT_DIR / "batch_summary.txt"

    env = os.environ.copy()

    # 保证 demo.py 能正常 import models.full_model
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    # 避免中文路径 / 中文输出乱码
    env["PYTHONIOENCODING"] = "utf-8"

    total_start = time.time()
    success_count = 0
    fail_count = 0
    time_list = []

    print("=" * 80)
    print("YOLOv11x 批量显著性排序开始")
    print("项目目录:", PROJECT_ROOT)
    print("使用 Python:", PYTHON_EXE)
    print("主脚本:", SCRIPT_PATH)
    print("图片目录:", IMG_ROOT)
    print("输出目录:", OUT_DIR)
    print("本次处理数量:", len(img_list))
    print("=" * 80)

    with open(runtime_log_path, "w", encoding="utf-8") as flog, \
         open(error_log_path, "w", encoding="utf-8") as ferr:

        flog.write("YOLOv11x Batch Runtime Log\n")
        flog.write("=" * 80 + "\n")
        flog.write(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
        flog.write(f"PYTHON_EXE: {PYTHON_EXE}\n")
        flog.write(f"SCRIPT_PATH: {SCRIPT_PATH}\n")
        flog.write(f"IMG_ROOT: {IMG_ROOT}\n")
        flog.write(f"OUT_DIR: {OUT_DIR}\n")
        flog.write(f"TOTAL_IMAGES: {len(img_list)}\n\n")

        for idx, img_path in enumerate(img_list, start=1):
            print("-" * 80)
            print(f"[{idx:03d}/{len(img_list):03d}] 正在处理: {img_path.name}")

            start = time.time()

            cmd = [
                str(PYTHON_EXE),
                str(SCRIPT_PATH),
                str(img_path)
            ]

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore"
                )

                elapsed = time.time() - start
                time_list.append(elapsed)
                success_count += 1

                print(f"[完成] {img_path.name} | 耗时: {elapsed:.4f}s")

                flog.write(f"[SUCCESS] {img_path.name}\n")
                flog.write(f"TIME: {elapsed:.4f}s\n")
                flog.write("STDOUT:\n")
                flog.write(result.stdout + "\n")

                if result.stderr.strip():
                    flog.write("STDERR:\n")
                    flog.write(result.stderr + "\n")

                flog.write("-" * 80 + "\n")

            except subprocess.CalledProcessError as e:
                elapsed = time.time() - start
                fail_count += 1

                print(f"[失败] {img_path.name} | 耗时: {elapsed:.4f}s")
                print("错误信息已经写入:", error_log_path)

                ferr.write(f"[FAILED] {img_path.name}\n")
                ferr.write(f"TIME: {elapsed:.4f}s\n")
                ferr.write("STDOUT:\n")
                ferr.write((e.stdout or "") + "\n")
                ferr.write("STDERR:\n")
                ferr.write((e.stderr or "") + "\n")
                ferr.write("-" * 80 + "\n")

                if not CONTINUE_ON_ERROR:
                    break

    total_time = time.time() - total_start
    avg_time = total_time / success_count if success_count > 0 else 0.0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("YOLOv11x Batch Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total images: {len(img_list)}\n")
        f.write(f"Success count: {success_count}\n")
        f.write(f"Fail count: {fail_count}\n")
        f.write(f"Total runtime: {total_time:.4f}s\n")
        f.write(f"Average runtime per success image: {avg_time:.4f}s\n")

        if len(time_list) > 0:
            f.write(f"Min runtime: {min(time_list):.4f}s\n")
            f.write(f"Max runtime: {max(time_list):.4f}s\n")

    print("=" * 80)
    print("批量处理完成")
    print("成功:", success_count)
    print("失败:", fail_count)
    print(f"总耗时: {total_time:.4f}s")
    print(f"平均每张耗时: {avg_time:.4f}s")
    print("运行日志:", runtime_log_path)
    print("错误日志:", error_log_path)
    print("汇总文件:", summary_path)
    print("=" * 80)


if __name__ == "__main__":
    main()