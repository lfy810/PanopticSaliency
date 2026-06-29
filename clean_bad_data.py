import os
import shutil
from PIL import Image
import numpy as np


# ============ 配置区 ============
DATA_ROOT = "data/F-360iSOD"

IMG_ROOT = os.path.join(DATA_ROOT, "stimulis")
OBJ_ROOT = os.path.join(DATA_ROOT, "objects")
INST_ROOT = os.path.join(DATA_ROOT, "instances")

BACKUP_ROOT = os.path.join(DATA_ROOT, "_bad_samples_backup")

# 小于这个面积就认为是无效标注
MIN_MASK_AREA = 50

# True = 复制脏数据到备份文件夹后，从原目录删除
# False = 只备份，不删除（推荐先用 False）
DELETE_BAD_FILES = True
# ===============================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def mask_area(mask_path):
    if not os.path.exists(mask_path):
        return 0

    try:
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        return int((mask_np > 0).sum())
    except Exception:
        return 0


def find_file_by_stem(folder, stem):
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    for ext in exts:
        path = os.path.join(folder, stem + ext)
        if os.path.exists(path):
            return path
    return None


def find_instance_files(inst_root, stem):
    """
    兼容两种情况：
    1. instances/001.png
    2. instances/001/xxx.png
    """
    files = []

    direct = find_file_by_stem(inst_root, stem)
    if direct is not None:
        files.append(direct)

    sub_dir = os.path.join(inst_root, stem)
    if os.path.isdir(sub_dir):
        for name in os.listdir(sub_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                files.append(os.path.join(sub_dir, name))

    return files


def total_instance_area(inst_files):
    total = 0
    for p in inst_files:
        total += mask_area(p)
    return total


def backup_file(src_path, backup_subdir):
    if src_path is None or not os.path.exists(src_path):
        return

    ensure_dir(backup_subdir)
    dst_path = os.path.join(backup_subdir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)

    if DELETE_BAD_FILES:
        os.remove(src_path)


def backup_dir(src_dir, backup_subdir):
    if src_dir is None or not os.path.isdir(src_dir):
        return

    ensure_dir(os.path.dirname(backup_subdir))

    if os.path.exists(backup_subdir):
        shutil.rmtree(backup_subdir)

    shutil.copytree(src_dir, backup_subdir)

    if DELETE_BAD_FILES:
        shutil.rmtree(src_dir)


def main():
    ensure_dir(BACKUP_ROOT)

    image_names = sorted([
        f for f in os.listdir(IMG_ROOT)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    bad_samples = []

    print("开始扫描数据集...")
    print("图片数量:", len(image_names))
    print("最小有效 mask 面积:", MIN_MASK_AREA)
    print("删除模式:", DELETE_BAD_FILES)
    print("-" * 60)

    for img_name in image_names:
        stem = os.path.splitext(img_name)[0]

        img_path = os.path.join(IMG_ROOT, img_name)
        obj_path = find_file_by_stem(OBJ_ROOT, stem)
        inst_files = find_instance_files(INST_ROOT, stem)

        obj_area = mask_area(obj_path) if obj_path else 0
        inst_area = total_instance_area(inst_files)

        bad_obj = obj_area < MIN_MASK_AREA
        bad_inst = inst_area < MIN_MASK_AREA

        # objects 和 instances 都无效，才认定为脏样本
        if bad_obj and bad_inst:
            bad_samples.append({
                "name": img_name,
                "stem": stem,
                "obj_area": obj_area,
                "inst_area": inst_area,
                "obj_path": obj_path,
                "inst_files": inst_files,
                "img_path": img_path
            })

            sample_backup_dir = os.path.join(BACKUP_ROOT, stem)

            backup_file(img_path, sample_backup_dir)

            if obj_path:
                backup_file(obj_path, sample_backup_dir)

            for inst_file in inst_files:
                backup_file(inst_file, sample_backup_dir)

            inst_sub_dir = os.path.join(INST_ROOT, stem)
            if os.path.isdir(inst_sub_dir):
                backup_dir(inst_sub_dir, os.path.join(sample_backup_dir, "instances"))

            print(
                f"[BAD] {img_name}  "
                f"object_area={obj_area}  instance_area={inst_area}"
            )

    report_path = os.path.join(BACKUP_ROOT, "bad_samples_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("F-360iSOD bad samples report\n")
        f.write(f"MIN_MASK_AREA = {MIN_MASK_AREA}\n")
        f.write(f"DELETE_BAD_FILES = {DELETE_BAD_FILES}\n")
        f.write("-" * 60 + "\n")

        for item in bad_samples:
            f.write(
                f"{item['name']}  "
                f"object_area={item['obj_area']}  "
                f"instance_area={item['inst_area']}\n"
            )

    print("\n========== 清洗完成 ==========")
    print("发现脏样本数量:", len(bad_samples))
    print("报告文件:", report_path)
    print("备份目录:", BACKUP_ROOT)

    if DELETE_BAD_FILES:
        print("已从原目录删除脏数据")
    else:
        print("当前仅备份，没有删除原文件")
        print("确认无误后，可把 DELETE_BAD_FILES 改成 True 再运行")


if __name__ == "__main__":
    main()