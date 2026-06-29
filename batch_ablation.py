import os
import re
import csv
import time
import subprocess
from pathlib import Path


# =========================
# 配置区
# =========================

PROJECT_ROOT = Path(r"D:\PanopticSaliency")

PYTHON_EXE = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
SCRIPT_PATH = PROJECT_ROOT / "demo_ablation_ready.py"

IMG_ROOT = PROJECT_ROOT / "data" / "F-360iSOD-test" / "stimulis"
OUT_ROOT = PROJECT_ROOT / "outputs_ablation"

MAX_IMAGES = None

IMG_SUFFIX = [".png", ".jpg", ".jpeg", ".bmp"]

CONTINUE_ON_ERROR = True


# =========================
# 消融实验组
# =========================

VARIANTS = [
    {
        "name": "full",
        "args": []
    },
    {
        "name": "no_supplement",
        "args": ["--disable_supplement"]
    },
    {
        "name": "no_attention",
        "args": ["--disable_attention"]
    },
    {
        "name": "no_nms",
        "args": ["--disable_nms"]
    },
]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_image_list(img_root: Path):
    img_list = []
    for p in img_root.iterdir():
        if p.suffix.lower() in IMG_SUFFIX:
            img_list.append(p)
    return sorted(img_list, key=lambda x: x.name)


def search_number(pattern, text, default=None, cast=float):
    m = re.search(pattern, text)
    if m:
        try:
            return cast(m.group(1))
        except Exception:
            return default
    return default


def parse_stdout(stdout):
    """
    从 demo_ablation_ready.py 输出中提取关键结果。
    """
    data = {}

    data["device"] = search_number(r"使用设备:\s*(\w+)", stdout, default="", cast=str)
    data["region_type"] = search_number(r"区域来源:\s*(.+)", stdout, default="", cast=str)

    data["yolo_count"] = search_number(r"YOLO原始候选区域数量:\s*(\d+)", stdout, default=0, cast=int)
    data["supplement_count"] = search_number(r"显著图补充区域数量:\s*(\d+)", stdout, default=0, cast=int)
    data["candidate_count"] = search_number(r"检测到候选区域数量:\s*(\d+)", stdout, default=0, cast=int)
    data["output_count"] = search_number(r"筛选出的重要区域数量:\s*(\d+)", stdout, default=0, cast=int)

    rank1_match = re.search(
        r"Rank1:\s*Region\s+(\d+)\s+score=([0-9.]+)\s+class=([a-zA-Z_]+)\s+yolo=([0-9.]+)\s+mean=([0-9.]+)",
        stdout
    )

    if rank1_match:
        data["rank1_region"] = int(rank1_match.group(1))
        data["rank1_score"] = float(rank1_match.group(2))
        data["rank1_class"] = rank1_match.group(3)
        data["rank1_yolo"] = float(rank1_match.group(4))
        data["rank1_mean"] = float(rank1_match.group(5))
    else:
        data["rank1_region"] = ""
        data["rank1_score"] = ""
        data["rank1_class"] = ""
        data["rank1_yolo"] = ""
        data["rank1_mean"] = ""

    return data


def main():
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"找不到虚拟环境 Python: {PYTHON_EXE}")

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"找不到主脚本: {SCRIPT_PATH}")

    if not IMG_ROOT.exists():
        raise FileNotFoundError(f"找不到图片目录: {IMG_ROOT}")

    ensure_dir(OUT_ROOT)

    img_list = get_image_list(IMG_ROOT)

    if MAX_IMAGES is not None:
        img_list = img_list[:MAX_IMAGES]

    if len(img_list) == 0:
        raise RuntimeError("没有找到可处理图片")

    summary_csv = OUT_ROOT / "ablation_summary.csv"
    log_txt = OUT_ROOT / "ablation_runtime_log.txt"
    error_txt = OUT_ROOT / "ablation_error_log.txt"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"

    rows = []

    total_start = time.time()

    print("=" * 80)
    print("开始消融实验")
    print("图片数量:", len(img_list))
    print("实验组:", [v["name"] for v in VARIANTS])
    print("输出目录:", OUT_ROOT)
    print("=" * 80)

    with open(log_txt, "w", encoding="utf-8") as flog, \
         open(error_txt, "w", encoding="utf-8") as ferr:

        for variant in VARIANTS:
            variant_name = variant["name"]
            variant_out_dir = OUT_ROOT / variant_name
            ensure_dir(variant_out_dir)

            print("\n" + "=" * 80)
            print(f"当前实验组: {variant_name}")
            print("=" * 80)

            for idx, img_path in enumerate(img_list, start=1):
                print(f"[{variant_name}] [{idx}/{len(img_list)}] {img_path.name}")

                start = time.time()

                cmd = [
                    str(PYTHON_EXE),
                    str(SCRIPT_PATH),
                    str(img_path),
                    "--out_dir",
                    str(variant_out_dir),
                    "--variant",
                    variant_name
                ] + variant["args"]

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
                    parsed = parse_stdout(result.stdout)

                    row = {
                        "variant": variant_name,
                        "image": img_path.name,
                        "runtime": round(elapsed, 4),
                        **parsed
                    }
                    rows.append(row)

                    print(
                        f"完成 | time={elapsed:.4f}s | "
                        f"rank1={parsed.get('rank1_class')} | "
                        f"score={parsed.get('rank1_score')}"
                    )

                    flog.write(f"[SUCCESS] {variant_name} | {img_path.name}\n")
                    flog.write(result.stdout + "\n")
                    if result.stderr.strip():
                        flog.write("STDERR:\n")
                        flog.write(result.stderr + "\n")
                    flog.write("-" * 80 + "\n")

                except subprocess.CalledProcessError as e:
                    elapsed = time.time() - start
                    print(f"失败 | {variant_name} | {img_path.name} | time={elapsed:.4f}s")

                    ferr.write(f"[FAILED] {variant_name} | {img_path.name}\n")
                    ferr.write("CMD:\n")
                    ferr.write(" ".join(cmd) + "\n")
                    ferr.write("STDOUT:\n")
                    ferr.write((e.stdout or "") + "\n")
                    ferr.write("STDERR:\n")
                    ferr.write((e.stderr or "") + "\n")
                    ferr.write("-" * 80 + "\n")

                    if not CONTINUE_ON_ERROR:
                        raise

    fieldnames = [
        "variant",
        "image",
        "runtime",
        "device",
        "region_type",
        "yolo_count",
        "supplement_count",
        "candidate_count",
        "output_count",
        "rank1_region",
        "rank1_class",
        "rank1_yolo",
        "rank1_score",
        "rank1_mean"
    ]

    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - total_start

    print("=" * 80)
    print("消融实验完成")
    print(f"总耗时: {total_time:.4f}s")
    print("结果汇总:", summary_csv)
    print("运行日志:", log_txt)
    print("错误日志:", error_txt)
    print("=" * 80)


if __name__ == "__main__":
    main()