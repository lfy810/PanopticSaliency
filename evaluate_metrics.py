import os
import torch
import numpy as np
import cv2

from datasets.panorama_train_dataset import PanoramaTrainDataset
from models.full_model import DistortionAwareSaliencyModel


# ============ 配置区 ============
IMG_ROOT = 'data/F-360iSOD-test/stimulis'
OBJ_ROOT = 'data/F-360iSOD-test/objects'
IMG_SIZE = (512, 1024)

MODEL_PATH = 'checkpoints/best_model_sgdaf_v7.pth'

N_SAMPLES = 100
MIN_AREA = 50
NUM_THRESHOLDS = 256
# ===============================


def compute_mae(pred, gt):
    return np.mean(np.abs(pred - gt))


def compute_f_at_threshold(pred, gt, thresh):
    pred_bin = (pred >= thresh).astype(np.float32)

    tp = (pred_bin * gt).sum()
    precision = tp / (pred_bin.sum() + 1e-8)
    recall = tp / (gt.sum() + 1e-8)

    beta2 = 0.3
    f = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return float(f)


def compute_all_fmeasures(pred, gt, num_thresholds=256):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    f_list = []

    for thresh in thresholds:
        f = compute_f_at_threshold(pred, gt, thresh)
        f_list.append(f)

    f_arr = np.array(f_list, dtype=np.float32)

    max_f = float(np.max(f_arr))
    avg_f = float(np.mean(f_arr))
    med_f = float(np.median(f_arr))
    min_f = float(np.min(f_arr))

    return max_f, avg_f, med_f, min_f


def compute_auc(pred, gt, num_thresholds=256):
    gt_bin = (gt > 0.5).astype(np.uint8)
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        pred_bin = (pred >= thresh).astype(np.uint8)

        tp = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
        tn = np.logical_and(pred_bin == 0, gt_bin == 0).sum()
        fp = np.logical_and(pred_bin == 1, gt_bin == 0).sum()
        fn = np.logical_and(pred_bin == 0, gt_bin == 1).sum()

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    tpr_arr = tpr_arr[order]

    auc = np.trapz(tpr_arr, fpr_arr)
    return float(auc)


def get_connected_regions(gt_mask, min_area=50):
    gt_bin = (gt_mask > 0.5).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(gt_bin)

    regions = []
    for region_id in range(1, num_labels):
        region = (labels == region_id)
        if region.sum() < min_area:
            continue
        regions.append(region)

    return regions


def compute_center_score(mask, h, w):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0

    cx = float(xs.mean())
    cy = float(ys.mean())

    img_center_x = w / 2.0
    img_center_y = h / 2.0
    max_center_dist = np.sqrt(img_center_x ** 2 + img_center_y ** 2)

    center_dist = np.sqrt((cx - img_center_x) ** 2 + (cy - img_center_y) ** 2)
    center_score = 1.0 - (center_dist / (max_center_dist + 1e-8))
    return float(center_score)


def compute_region_prior_scores(regions, h, w):
    area_scores = []
    center_scores = []

    image_area = h * w

    for region in regions:
        area_ratio = float(region.sum()) / float(image_area)
        area_score = np.sqrt(area_ratio)
        center_score = compute_center_score(region, h, w)

        area_scores.append(area_score)
        center_scores.append(center_score)

    area_scores = np.array(area_scores, dtype=np.float32)
    center_scores = np.array(center_scores, dtype=np.float32)

    def norm(x):
        if len(x) == 0:
            return x
        x_min = x.min()
        x_max = x.max()
        if abs(x_max - x_min) < 1e-8:
            return np.ones_like(x) * 0.5
        return (x - x_min) / (x_max - x_min + 1e-8)

    area_norm = norm(area_scores)
    center_norm = norm(center_scores)

    prior = 0.7 * area_norm + 0.3 * center_norm
    return prior


def compute_pra(pred_map, gt_mask, min_area=50):
    h, w = gt_mask.shape
    regions = get_connected_regions(gt_mask, min_area)

    if len(regions) < 2:
        return None, None

    pred_scores = []
    for region in regions:
        pred_scores.append(float(pred_map[region].mean()))

    gt_prior = compute_region_prior_scores(regions, h, w)

    total_pairs = 0
    correct_pairs = 0

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if abs(gt_prior[i] - gt_prior[j]) < 1e-8:
                continue

            gt_order = gt_prior[i] > gt_prior[j]
            pred_order = pred_scores[i] > pred_scores[j]

            if gt_order == pred_order:
                correct_pairs += 1

            total_pairs += 1

    if total_pairs == 0:
        return None, None

    pra = correct_pairs / (total_pairs + 1e-8)

    pred_top1 = int(np.argmax(pred_scores))
    gt_top1 = int(np.argmax(gt_prior))
    top1_consistency = 1.0 if pred_top1 == gt_top1 else 0.0

    return float(pra), float(top1_consistency)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('使用设备:', device)

    dataset = PanoramaTrainDataset(
        img_root=IMG_ROOT,
        obj_root=OBJ_ROOT,
        img_size=IMG_SIZE
    )

    model = DistortionAwareSaliencyModel(img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    n = min(N_SAMPLES, len(dataset))

    mae_list = []
    auc_list = []

    max_f_list = []
    avg_f_list = []
    med_f_list = []
    min_f_list = []

    pra_list = []
    top1_list = []

    skipped_empty = 0
    skipped_region = 0
    valid_count = 0

    print(f'评估样本数: {n}')
    print('----------------------------------')

    with torch.no_grad():
        for i in range(n):
            image, mask, name = dataset[i]

            image_gpu = image.unsqueeze(0).to(device)
            logits = model(image_gpu)
            pred = torch.sigmoid(logits).squeeze().cpu().numpy()

            gt = mask.squeeze().cpu().numpy()

            if gt.sum() < MIN_AREA:
                skipped_empty += 1
                print(f'[{i+1:03d}/{n:03d}] {name}  跳过（mask为空或太小）')
                continue

            regions = get_connected_regions(gt, MIN_AREA)
            if len(regions) < 2:
                skipped_region += 1
                print(f'[{i+1:03d}/{n:03d}] {name}  跳过（有效区域不足）')
                continue

            mae = compute_mae(pred, gt)
            auc = compute_auc(pred, gt, NUM_THRESHOLDS)
            max_f, avg_f, med_f, min_f = compute_all_fmeasures(pred, gt, NUM_THRESHOLDS)

            pra, top1 = compute_pra(pred, gt, MIN_AREA)
            if pra is None:
                skipped_region += 1
                print(f'[{i+1:03d}/{n:03d}] {name}  跳过（PRA无法计算）')
                continue

            mae_list.append(mae)
            auc_list.append(auc)

            max_f_list.append(max_f)
            avg_f_list.append(avg_f)
            med_f_list.append(med_f)
            min_f_list.append(min_f)

            pra_list.append(pra)
            top1_list.append(top1)

            valid_count += 1

            print(
                f'[{i+1:03d}/{n:03d}] {name}  '
                f'MAE={mae:.4f}  '
                f'AUC={auc:.4f}  '
                f'maxF={max_f:.4f}  '
                f'PRA={pra:.4f}  '
                f'Top1={top1:.0f}'
            )

    print('\n========== 统计结果 ==========')
    print(f'总样本数: {n}')
    print(f'有效样本数: {valid_count}')
    print(f'跳过（mask为空或太小）: {skipped_empty}')
    print(f'跳过（有效区域不足）: {skipped_region}')

    if valid_count > 0:
        print('\n========== 平均结果 ==========')
        print(f'MAE ↓ : {np.mean(mae_list):.4f}')
        print(f'AUC ↑ : {np.mean(auc_list):.4f}')
        print(f'max F-measure ↑ : {np.mean(max_f_list):.4f}')
        print(f'avg F-measure ↑ : {np.mean(avg_f_list):.4f}')
        print(f'med F-measure ↑ : {np.mean(med_f_list):.4f}')
        print(f'min F-measure ↑ : {np.mean(min_f_list):.4f}')
        print(f'PRA ↑ : {np.mean(pra_list):.4f}')
        print(f'Top1 Consistency ↑ : {np.mean(top1_list):.4f}')
    else:
        print('\n没有可用于统计的有效样本')


if __name__ == '__main__':
    main()