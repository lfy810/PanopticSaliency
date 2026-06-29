import os
import glob
import numpy as np
from PIL import Image


def load_instance_masks(inst_root, name, target_size=(512, 1024), min_area=50):
    # name 例如 001.png
    stem = os.path.splitext(name)[0]

    # 先试 flat 标注图：instances/001.png
    flat_path = os.path.join(inst_root, name)
    if os.path.exists(flat_path):
        return _load_from_single_label_map(flat_path, target_size, min_area)

    # 再试文件夹：instances/001/*.png
    folder_path = os.path.join(inst_root, stem)
    if os.path.isdir(folder_path):
        return _load_from_multi_mask_folder(folder_path, target_size, min_area)

    # 再试 instances 下带 stem 的多文件
    pattern = os.path.join(inst_root, f'{stem}*.png')
    files = sorted(glob.glob(pattern))
    if len(files) > 0:
        return _load_from_multi_mask_files(files, target_size, min_area)

    return []


def _load_from_single_label_map(path, target_size, min_area):
    # 这种情况是一张图里不同像素值代表不同 instance id
    mask_img = Image.open(path).convert('L')
    mask_img = mask_img.resize((target_size[1], target_size[0]), resample=Image.NEAREST)
    mask = np.array(mask_img)

    ids = np.unique(mask)
    ids = ids[ids > 0]

    masks = []
    for instance_id in ids:
        m = (mask == instance_id).astype(np.uint8)
        if m.sum() < min_area:
            continue
        masks.append(m)

    return masks


def _load_from_multi_mask_folder(folder_path, target_size, min_area):
    files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    return _load_from_multi_mask_files(files, target_size, min_area)


def _load_from_multi_mask_files(files, target_size, min_area):
    masks = []

    for path in files:
        mask_img = Image.open(path).convert('L')
        mask_img = mask_img.resize((target_size[1], target_size[0]), resample=Image.NEAREST)
        mask = np.array(mask_img)

        mask = (mask > 0).astype(np.uint8)
        if mask.sum() < min_area:
            continue

        masks.append(mask)

    return masks