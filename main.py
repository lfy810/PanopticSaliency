import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'utils'))

import time
import numpy as np
import torch
import cv2

from datasets.panorama_dataset import PanoramaDataset
from models.full_model import DistortionAwareSaliencyModel
from instance_loader import load_instance_masks


# ============ 配置区 ============
IMG_ROOT = 'data/F-360iSOD/stimulis'
OBJ_ROOT = 'data/F-360iSOD/objects'
INST_ROOT = 'data/F-360iSOD/instances'
IMG_SIZE = (512, 1024)

MODEL_PATH = 'checkpoints/best_model.pth'

N_SAMPLES = 83
TOPK = 3
MIN_AREA = 220
IOU_THRESH = 0.40

OUT_DIR = 'outputs'
TXT_PATH = os.path.join(OUT_DIR, 'instance_ranking_final_best.txt')
VIS_DIR = os.path.join(OUT_DIR, 'vis_instance_final_best')
# ===============================


def minmax_norm(values):
    values = np.array(values, dtype=np.float32)
    if len(values) == 0:
        return values

    v_min = values.min()
    v_max = values.max()

    if abs(v_max - v_min) < 1e-8:
        return np.ones_like(values) * 0.5

    return (values - v_min) / (v_max - v_min + 1e-8)


def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return inter / (union + 1e-8)


def extract_object_masks(obj_mask_pil, min_area=220):
    m = np.array(obj_mask_pil)
    bin_mask = (m > 0).astype(np.uint8) * 255

    num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)

    masks = []
    for lab in range(1, num_labels):
        region = (labels == lab).astype(np.uint8)
        if region.sum() < min_area:
            continue
        masks.append(region)

    return masks


def build_candidate_regions(name, obj_mask_pil):
    instance_masks = load_instance_masks(
        INST_ROOT,
        name,
        target_size=IMG_SIZE,
        min_area=MIN_AREA
    )

    if len(instance_masks) > 0:
        return instance_masks, 'instance'

    object_masks = extract_object_masks(obj_mask_pil, MIN_AREA)
    return object_masks, 'object'


def compute_region_features(image_tensor, saliency_map, region_masks):
    s = saliency_map.squeeze(0).squeeze(0).detach().cpu().numpy()

    img = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)

    h, w, _ = img.shape
    image_area = h * w

    img_center_x = w / 2.0
    img_center_y = h / 2.0
    max_center_dist = np.sqrt(img_center_x ** 2 + img_center_y ** 2)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    global_gray_mean = float(gray.mean())

    mean_sal_list = []
    max_sal_list = []
    area_list = []
    color_list = []
    center_list = []
    contrast_list = []
    edge_list = []

    for mask in region_masks:
        region = mask.astype(bool)

        region_sal = s[region]
        region_sat = sat[region]
        region_val = val[region]
        region_gray = gray[region]

        mean_sal = float(region_sal.mean()) if region_sal.size > 0 else 0.0
        max_sal = float(region_sal.max()) if region_sal.size > 0 else 0.0

        area_ratio = float(region.sum()) / float(image_area)
        area_score = np.sqrt(area_ratio)

        color_score = 0.5 * float(region_sat.mean()) + 0.5 * float(region_val.mean())

        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cx = float(xs.mean())
            cy = float(ys.mean())

            center_dist = np.sqrt((cx - img_center_x) ** 2 + (cy - img_center_y) ** 2)
            center_score = 1.0 - (center_dist / (max_center_dist + 1e-8))

            left_gap = cx / (w + 1e-8)
            right_gap = (w - cx) / (w + 1e-8)
            top_gap = cy / (h + 1e-8)
            bottom_gap = (h - cy) / (h + 1e-8)

            edge_gap = min(left_gap, right_gap, top_gap, bottom_gap)
            edge_penalty_base = edge_gap
        else:
            center_score = 0.0
            edge_penalty_base = 0.0

        region_gray_mean = float(region_gray.mean()) if region_gray.size > 0 else 0.0
        contrast_score = abs(region_gray_mean - global_gray_mean)

        mean_sal_list.append(mean_sal)
        max_sal_list.append(max_sal)
        area_list.append(area_score)
        color_list.append(color_score)
        center_list.append(center_score)
        contrast_list.append(contrast_score)
        edge_list.append(edge_penalty_base)

    mean_sal_norm = minmax_norm(mean_sal_list)
    max_sal_norm = minmax_norm(max_sal_list)
    area_norm = minmax_norm(area_list)
    color_norm = minmax_norm(color_list)
    center_norm = minmax_norm(center_list)
    contrast_norm = minmax_norm(contrast_list)
    edge_norm = minmax_norm(edge_list)

    features = []
    for i in range(len(region_masks)):
        area_score = float(area_list[i])

        penalty = 1.0
        if area_score < 0.050:
            penalty *= 0.78
        if area_score < 0.030:
            penalty *= 0.60

        edge_factor = 0.90 + 0.10 * float(edge_norm[i])

        features.append({
            'mean_sal': float(mean_sal_list[i]),
            'max_sal': float(max_sal_list[i]),
            'area_score': float(area_list[i]),
            'color_score': float(color_list[i]),
            'center_score': float(center_list[i]),
            'contrast_score': float(contrast_list[i]),

            'mean_sal_norm': float(mean_sal_norm[i]),
            'max_sal_norm': float(max_sal_norm[i]),
            'area_norm': float(area_norm[i]),
            'color_norm': float(color_norm[i]),
            'center_norm': float(center_norm[i]),
            'contrast_norm': float(contrast_norm[i]),
            'edge_norm': float(edge_norm[i]),

            'penalty': float(penalty),
            'edge_factor': float(edge_factor)
        })

    return features


def rank_regions(image_tensor, saliency_map, region_masks):
    features = compute_region_features(image_tensor, saliency_map, region_masks)

    scores = []
    for i, feat in enumerate(features):
        base_score = (
            0.22 * feat['mean_sal_norm'] +
            0.08 * feat['max_sal_norm'] +
            0.28 * feat['area_norm'] +
            0.08 * feat['color_norm'] +
            0.20 * feat['center_norm'] +
            0.14 * feat['contrast_norm']
        )

        final_score = base_score * feat['penalty'] * feat['edge_factor']
        scores.append((i, float(final_score), feat))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def suppress_overlaps(ranking, region_masks, topk=3, iou_thresh=0.4):
    selected = []

    for item in ranking:
        idx, score, feat = item
        keep = True

        for sel in selected:
            sel_idx = sel[0]
            iou = compute_iou(region_masks[idx], region_masks[sel_idx])
            if iou > iou_thresh:
                keep = False
                break

        if keep:
            selected.append(item)

        if len(selected) >= topk:
            break

    return selected


def draw_topk_on_image(img_tensor, region_masks, ranking, save_path):
    img = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for rank_id, (region_id, score, feat) in enumerate(ranking):
        mask = region_masks[region_id]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if rank_id == 0:
            color = (0, 0, 255)
        elif rank_id == 1:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.drawContours(img, contours, -1, color, 2)

        ys, xs = np.where(mask == 1)
        if len(xs) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())

            text1 = f'{rank_id + 1}:{score:.3f}'
            text2 = f'm:{feat["mean_sal"]:.2f} a:{feat["area_score"]:.2f}'
            text3 = f'ctr:{feat["center_score"]:.2f} con:{feat["contrast_score"]:.2f}'

            cv2.putText(img, text1, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, text2, (cx, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(img, text3, (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(save_path, img)


def save_text_result(name, region_type, ranking):
    with open(TXT_PATH, 'a', encoding='utf-8') as f:
        f.write(f'\n{name}  ({region_type})\n')
        for region_id, score, feat in ranking:
            f.write(
                f'  Region {region_id:02d}  '
                f'score={score:.6f}  '
                f'mean={feat["mean_sal"]:.6f}  '
                f'max={feat["max_sal"]:.6f}  '
                f'area={feat["area_score"]:.6f}  '
                f'color={feat["color_score"]:.6f}  '
                f'center={feat["center_score"]:.6f}  '
                f'contrast={feat["contrast_score"]:.6f}  '
                f'penalty={feat["penalty"]:.6f}  '
                f'edge={feat["edge_factor"]:.6f}\n'
            )


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('使用设备:', device)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    with open(TXT_PATH, 'w', encoding='utf-8') as f:
        f.write('final optimized ranking results\n')
        f.write(f'img_root={IMG_ROOT}\n')
        f.write(f'inst_root={INST_ROOT}\n')
        f.write(f'obj_root={OBJ_ROOT}\n')
        f.write(f'img_size={IMG_SIZE}\n')
        f.write(f'n_samples={N_SAMPLES}, topk={TOPK}, min_area={MIN_AREA}, iou_thresh={IOU_THRESH}\n')
        f.write('score = (0.22*mean + 0.08*max + 0.28*area + 0.08*color + 0.20*center + 0.14*contrast) * penalty * edge_factor\n')
        f.write('fallback: instance -> object\n')
        f.write('-' * 110 + '\n')

    dataset = PanoramaDataset(
        img_root=IMG_ROOT,
        obj_root=OBJ_ROOT,
        img_size=IMG_SIZE
    )

    total = len(dataset)
    n = min(N_SAMPLES, total)
    print('数据集大小:', total, '本次遍历:', n)

    model = DistortionAwareSaliencyModel(img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    valid_count = 0

    for idx in range(n):
        image, obj_mask_pil, name = dataset[idx]
        image_gpu = image.unsqueeze(0).to(device)

        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            saliency_map = model(image_gpu)
            saliency_map = torch.sigmoid(saliency_map)

        if device == 'cuda':
            torch.cuda.synchronize()
        t_cost = time.time() - t0

        region_masks, region_type = build_candidate_regions(name, obj_mask_pil)

        if len(region_masks) == 0:
            print(f'[{idx+1:03d}/{n:03d}] {name}  没找到有效区域')
            continue

        ranking_all = rank_regions(image_gpu, saliency_map, region_masks)
        topk = suppress_overlaps(ranking_all, region_masks, topk=TOPK, iou_thresh=IOU_THRESH)

        save_text_result(name, region_type, topk)

        save_path = os.path.join(VIS_DIR, name.replace('.png', '_vis.png'))
        draw_topk_on_image(image_gpu, region_masks, topk, save_path)

        valid_count += 1
        print(
            f'[{idx+1:03d}/{n:03d}] {name}  '
            f'type={region_type}  '
            f'regions={len(region_masks)}  '
            f'time={t_cost:.3f}s  '
            f'saved={save_path}'
        )

    print('\n结果已保存到:')
    print('排序文本:', TXT_PATH)
    print('可视化图:', VIS_DIR)
    print('成功处理样本数:', valid_count)


if __name__ == '__main__':
    main()