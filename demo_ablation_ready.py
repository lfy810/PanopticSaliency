import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

import argparse
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms

from ultralytics import YOLO
from models.full_model import DistortionAwareSaliencyModel


# ============ 配置区 ============
IMG_SIZE = (512, 1024)   # (H, W)

MODEL_PATH = 'checkpoints/best_model_sgdaf_v7.pth'
YOLO_SEG_MODEL = 'yolo11x-seg.pt'

MIN_AREA = 100
YOLO_CONF = 0.15

# YOLO 区域不足 5 个时，从显著性图补足
MIN_YOLO_REGION_COUNT = 5
MAX_SALIENCY_SUPPLEMENT = 5
SUPPLEMENT_MIN_AREA = 30
SALIENCY_DUP_IOU_THRESH = 0.20

PERSON_SCORE_DECAY = 0.75
OBJECT_SCORE_BOOST = 1.12
SMALL_OBJECT_BOOST = 1.18
LARGE_OBJECT_BOOST = 1.20
LARGE_OBJECT_AREA_TH = 0.18
SALIENCY_REGION_BOOST = 1.18

IMPORTANCE_THRESHOLD = 0.25

ATTENTION_TEMPERATURE = 0.5
ATTENTION_ALPHA = 0.15

NMS_IOU_THRESH = 0.35
TOP_K_OUTPUT = 10

OUT_DIR = 'outputs_yolo11'
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


def softmax_np(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-8)


def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return inter / (union + 1e-8)


def merge_wraparound_regions(
    region_masks,
    class_names,
    conf_scores,
    edge_ratio=0.015,
    y_overlap_thresh=0.70,
    height_ratio_thresh=0.65,
    area_ratio_thresh=0.40,
    require_same_class=True
):
    if len(region_masks) <= 1:
        return region_masks, class_names, conf_scores

    h, w = region_masks[0].shape
    edge_width = max(2, int(w * edge_ratio))

    used = [False] * len(region_masks)

    new_masks = []
    new_names = []
    new_confs = []

    for i in range(len(region_masks)):
        if used[i]:
            continue

        mask_i = region_masks[i]
        ys_i, xs_i = np.where(mask_i > 0)

        if len(xs_i) == 0:
            used[i] = True
            continue

        touch_left_i = xs_i.min() <= edge_width
        touch_right_i = xs_i.max() >= w - edge_width

        merged_mask = mask_i.copy()
        merged_name = class_names[i]
        merged_conf = conf_scores[i]

        used[i] = True

        for j in range(i + 1, len(region_masks)):
            if used[j]:
                continue

            mask_j = region_masks[j]
            ys_j, xs_j = np.where(mask_j > 0)

            if len(xs_j) == 0:
                used[j] = True
                continue

            touch_left_j = xs_j.min() <= edge_width
            touch_right_j = xs_j.max() >= w - edge_width

            is_wrap_pair = (touch_left_i and touch_right_j) or (touch_right_i and touch_left_j)

            if not is_wrap_pair:
                continue

            if require_same_class and class_names[i] != class_names[j]:
                continue

            y1_i, y2_i = ys_i.min(), ys_i.max()
            y1_j, y2_j = ys_j.min(), ys_j.max()

            h_i = y2_i - y1_i + 1
            h_j = y2_j - y1_j + 1

            inter_y = max(0, min(y2_i, y2_j) - max(y1_i, y1_j))
            union_y = max(y2_i, y2_j) - min(y1_i, y1_j) + 1e-8
            y_overlap = inter_y / union_y

            height_ratio = min(h_i, h_j) / (max(h_i, h_j) + 1e-8)

            area_i = float(mask_i.sum())
            area_j = float(mask_j.sum())
            area_ratio = min(area_i, area_j) / (max(area_i, area_j) + 1e-8)

            if (
                y_overlap >= y_overlap_thresh
                and height_ratio >= height_ratio_thresh
                and area_ratio >= area_ratio_thresh
            ):
                merged_mask = np.logical_or(merged_mask > 0, mask_j > 0).astype(np.uint8)
                merged_conf = max(merged_conf, conf_scores[j])
                used[j] = True

        new_masks.append(merged_mask)
        new_names.append(merged_name)
        new_confs.append(merged_conf)

    return new_masks, new_names, new_confs


def read_and_preprocess_image(img_path, img_size=(512, 1024)):
    image_pil = Image.open(img_path).convert('RGB')
    image_resized = image_pil.resize((img_size[1], img_size[0]), Image.BILINEAR)
    image_tensor = transforms.ToTensor()(image_resized)
    return image_pil, image_tensor


def run_saliency_model(model, image_tensor, device):
    image_gpu = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        saliency_map = model(image_gpu)
        saliency_map = torch.sigmoid(saliency_map)
    return saliency_map


def resize_mask_to_target(mask, target_hw):
    target_h, target_w = target_hw
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    return (resized > 0).astype(np.uint8)


def run_yolo_segmentation(yolo_model, image_tensor, conf=0.15, min_area=100, target_hw=(512, 1024)):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    results = yolo_model.predict(
        source=img,
        conf=conf,
        verbose=False
    )

    if len(results) == 0:
        return [], [], []

    result = results[0]

    if result.masks is None or result.boxes is None:
        return [], [], []

    masks_data = result.masks.data.cpu().numpy()
    boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
    boxes_conf = result.boxes.conf.cpu().numpy()

    region_masks = []
    class_names = []
    conf_scores = []

    names_dict = result.names

    for i in range(len(masks_data)):
        raw_mask = (masks_data[i] > 0.5).astype(np.uint8)
        mask = resize_mask_to_target(raw_mask, target_hw)

        if mask.sum() < min_area:
            continue

        cls_id = int(boxes_cls[i])
        cls_name = names_dict.get(cls_id, str(cls_id))
        conf_score = float(boxes_conf[i])

        region_masks.append(mask)
        class_names.append(cls_name)
        conf_scores.append(conf_score)

    return region_masks, class_names, conf_scores


def add_saliency_supplement_regions(
    region_masks,
    class_names,
    conf_scores,
    pred_map,
    min_area=30,
    target_count=5,
    max_add=5,
    dup_iou_thresh=0.20
):
    """
    当 YOLO 候选区域不足 target_count 时：
    1. 从显著性热力图中提取候选区域；
    2. 排除 YOLO 已有区域；
    3. 剩余热力区域按面积从大到小排序；
    4. 补足到 target_count 个。
    """
    current_count = len(region_masks)
    need_add = max(0, target_count - current_count)

    if need_add <= 0:
        return region_masks, class_names, conf_scores, 0

    h, w = pred_map.shape

    occupied = np.zeros((h, w), dtype=np.uint8)

    for old_m in region_masks:
        if old_m.shape != pred_map.shape:
            old_m = cv2.resize(
                old_m.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
            old_m = (old_m > 0).astype(np.uint8)

        occupied = np.logical_or(occupied > 0, old_m > 0).astype(np.uint8)

    sal = pred_map.copy()
    sal_u8 = (sal * 255).astype(np.uint8)
    sal_u8 = cv2.GaussianBlur(sal_u8, (7, 7), 0)

    _, bin_mask = cv2.threshold(
        sal_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 去掉 YOLO 已经覆盖的区域
    bin_mask[occupied > 0] = 0

    kernel = np.ones((5, 5), np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)

    candidates = []

    for lab in range(1, num_labels):
        region = (labels == lab).astype(np.uint8)

        area = int(region.sum())
        if area < min_area:
            continue

        duplicate = False
        for old_m in region_masks:
            if compute_iou(region, old_m) > dup_iou_thresh:
                duplicate = True
                break

        if duplicate:
            continue

        mean_sal = float(pred_map[region.astype(bool)].mean()) if area > 0 else 0.0

        candidates.append({
            "mask": region,
            "area": area,
            "mean_sal": mean_sal
        })

    # 核心：按区域面积从大到小补足
    candidates.sort(key=lambda x: x["area"], reverse=True)

    add_num = min(need_add, max_add, len(candidates))

    for i in range(add_num):
        m = candidates[i]["mask"]
        region_masks.append(m.astype(np.uint8))
        class_names.append("saliency_region")
        conf_scores.append(0.50)

    return region_masks, class_names, conf_scores, add_num


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

    for i, mask in enumerate(region_masks):
        if mask.shape != s.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (s.shape[1], s.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            mask = (mask > 0).astype(np.uint8)
            region_masks[i] = mask

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

            'penalty': float(penalty),
            'edge_factor': float(edge_factor)
        })

    return features


def attention_refine_scores(base_scores, features, temperature=0.5, alpha=0.25):
    n = len(base_scores)

    if n <= 1:
        return base_scores

    feat_mat = []

    for feat in features:
        feat_vec = [
            feat['mean_sal_norm'],
            feat['max_sal_norm'],
            feat['area_norm'],
            feat['color_norm'],
            feat['center_norm'],
            feat['contrast_norm'],
            feat['edge_factor'],
        ]
        feat_mat.append(feat_vec)

    feat_mat = np.array(feat_mat, dtype=np.float32)
    base_scores_np = np.array(base_scores, dtype=np.float32)

    norm = np.linalg.norm(feat_mat, axis=1, keepdims=True) + 1e-8
    feat_norm = feat_mat / norm

    sim_matrix = np.matmul(feat_norm, feat_norm.T)

    refined_scores = []

    for i in range(n):
        sim = sim_matrix[i].copy()
        sim[i] = -1e9

        attn = softmax_np(sim / temperature)
        context_score = float(np.sum(attn * base_scores_np))

        refined_score = (1.0 - alpha) * base_scores_np[i] + alpha * context_score
        refined_scores.append(float(refined_score))

    return refined_scores


def rank_regions(image_tensor, saliency_map, region_masks, class_names=None, det_conf=None):
    features = compute_region_features(image_tensor, saliency_map, region_masks)

    if class_names is None:
        class_names = ['unknown'] * len(region_masks)

    if det_conf is None:
        det_conf = [0.5] * len(region_masks)

    base_scores = []

    for i, feat in enumerate(features):
        det_score = float(det_conf[i])
        cls_name = class_names[i].lower()

        base_score = (
            0.28 * feat['mean_sal_norm'] +
            0.10 * feat['max_sal_norm'] +
            0.18 * feat['area_norm'] +
            0.16 * feat['center_norm'] +
            0.16 * feat['contrast_norm'] +
            0.08 * feat['color_norm'] +
            0.08 * det_score
        )

        if cls_name == 'person':
            base_score *= PERSON_SCORE_DECAY
        else:
            base_score *= OBJECT_SCORE_BOOST

        if cls_name == 'saliency_region':
            base_score *= SALIENCY_REGION_BOOST

        if feat['area_score'] < 0.08 and feat['mean_sal_norm'] > 0.35:
            base_score *= SMALL_OBJECT_BOOST

        if feat['area_norm'] > LARGE_OBJECT_AREA_TH and feat['mean_sal_norm'] > 0.20:
            base_score *= LARGE_OBJECT_BOOST

        if feat['area_score'] < 0.01:
            base_score *= 0.75

        if feat['mean_sal'] < 0.05:
            base_score *= 0.90

        base_score = base_score * feat['penalty'] * feat['edge_factor']
        base_scores.append(float(base_score))

    refined_scores = attention_refine_scores(
        base_scores,
        features,
        temperature=ATTENTION_TEMPERATURE,
        alpha=ATTENTION_ALPHA
    )

    scores = []
    for i, feat in enumerate(features):
        scores.append(
            (
                i,
                float(refined_scores[i]),
                feat,
                class_names[i],
                float(det_conf[i])
            )
        )

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def select_important_regions(ranking, region_masks, top_k=10, iou_thresh=0.35):
    """
    固定输出 Top-K 重要区域，并使用 NMS 去除高度重叠区域。
    """
    selected = []

    for item in ranking:
        idx = item[0]

        keep = True

        for sel in selected:
            sel_idx = sel[0]
            iou = compute_iou(region_masks[idx], region_masks[sel_idx])

            if iou > iou_thresh:
                keep = False
                break

        if keep:
            selected.append(item)

        if len(selected) >= top_k:
            break

    if len(selected) == 0 and len(ranking) > 0:
        selected.append(ranking[0])

    return selected


def draw_saliency_overlay(image_tensor, pred_map, save_path):
    img = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pred = pred_map.squeeze().detach().cpu().numpy()
    heat = (pred * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    vis = cv2.addWeighted(img_bgr, 0.6, heat, 0.4, 0)
    cv2.imwrite(save_path, vis)


def smooth_mask_contour(mask, kernel_size=3):
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.GaussianBlur(mask_u8, (3, 3), 0)

    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    return (mask_u8 > 0).astype(np.uint8)


def get_mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_label(img_bgr, label, x1, y1, x2, y2, color):
    h, w, _ = img_bgr.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    font_thick = 1

    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thick)
    text_w, text_h = text_size

    tx = x1
    ty = y1 - 6

    if ty - text_h < 0:
        ty = y2 + text_h + 6

    if tx + text_w > w:
        tx = w - text_w - 3

    if tx < 0:
        tx = 3

    bg_x1 = max(tx - 3, 0)
    bg_y1 = max(ty - text_h - 4, 0)
    bg_x2 = min(tx + text_w + 3, w - 1)
    bg_y2 = min(ty + 4, h - 1)

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    img_bgr[:] = cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0)

    cv2.putText(img_bgr, label, (tx, ty), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, label, (tx, ty), font, font_scale, color, font_thick, cv2.LINE_AA)


def draw_ranking_result(image_tensor, region_masks, ranking, save_path):
    img = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (128, 0, 255),
        (255, 128, 0),
        (128, 255, 0)
    ]

    for rank_id, item in enumerate(ranking):
        region_id, score, feat, cls_name, det_score = item
        mask = region_masks[region_id]
        color = colors[rank_id % len(colors)]
        label = f'R{rank_id + 1}'

        bbox = get_mask_bbox(mask)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        if cls_name == 'saliency_region' or cls_name == 'pseudo_region':
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        else:
            smooth_mask = smooth_mask_contour(mask, kernel_size=3)

            contours, _ = cv2.findContours(
                smooth_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            area = smooth_mask.sum()

            if area < 3000:
                contour_thickness = 1
            elif area < 15000:
                contour_thickness = 1
            else:
                contour_thickness = 2

            cv2.drawContours(
                img_bgr,
                contours,
                -1,
                (0, 0, 0),
                contour_thickness + 1,
                lineType=cv2.LINE_AA
            )

            cv2.drawContours(
                img_bgr,
                contours,
                -1,
                color,
                contour_thickness,
                lineType=cv2.LINE_AA
            )

        draw_label(img_bgr, label, x1, y1, x2, y2, color)

    cv2.imwrite(save_path, img_bgr)


def save_text_result(image_name, ranking, region_type, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'image: {image_name}\n')
        f.write(f'region_type: {region_type}\n')
        f.write(f'importance_threshold: {IMPORTANCE_THRESHOLD}\n')
        f.write(f'attention_temperature: {ATTENTION_TEMPERATURE}\n')
        f.write(f'attention_alpha: {ATTENTION_ALPHA}\n')
        f.write(f'nms_iou_thresh: {NMS_IOU_THRESH}\n')
        f.write(f'min_yolo_region_count: {MIN_YOLO_REGION_COUNT}\n')
        f.write(f'supplement_min_area: {SUPPLEMENT_MIN_AREA}\n')
        f.write('-' * 100 + '\n')

        for rank_id, item in enumerate(ranking):
            region_id, score, feat, cls_name, det_score = item
            f.write(
                f'Rank{rank_id + 1}  Region {region_id:02d}  '
                f'class={cls_name}  '
                f'yolo_conf={det_score:.4f}  '
                f'final_score={score:.6f}  '
                f'mean={feat["mean_sal"]:.6f}  '
                f'max={feat["max_sal"]:.6f}  '
                f'area={feat["area_score"]:.6f}  '
                f'color={feat["color_score"]:.6f}  '
                f'center={feat["center_score"]:.6f}  '
                f'contrast={feat["contrast_score"]:.6f}\n'
            )


def main():
    """
    单张全景图显著性排序主程序。

    支持普通实验和消融实验：
    1. full：完整方法；
    2. --disable_supplement：关闭显著性区域补充；
    3. --disable_attention：关闭 Attention 融合；
    4. --disable_nms：关闭 NMS 去重。
    """
    global OUT_DIR, ATTENTION_ALPHA

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='输入全景图路径')

    # 消融实验相关参数
    parser.add_argument('--out_dir', type=str, default=OUT_DIR, help='输出目录')
    parser.add_argument('--variant', type=str, default='full', help='实验组名称')
    parser.add_argument('--disable_supplement', action='store_true', help='关闭显著性区域补充')
    parser.add_argument('--disable_attention', action='store_true', help='关闭 Attention 融合')
    parser.add_argument('--disable_nms', action='store_true', help='关闭 NMS 去重')

    args = parser.parse_args()

    # 根据命令行参数覆盖输出目录
    OUT_DIR = args.out_dir

    # Attention 消融：将融合强度设为 0，相当于只使用基础得分
    if args.disable_attention:
        ATTENTION_ALPHA = 0.0

    img_path = args.image_path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'找不到图片: {img_path}')

    os.makedirs(OUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('使用设备:', device)
    print('实验组:', args.variant)

    saliency_model = DistortionAwareSaliencyModel(img_size=IMG_SIZE).to(device)
    saliency_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    saliency_model.eval()

    yolo_model = YOLO(YOLO_SEG_MODEL)

    image_pil, image_tensor = read_and_preprocess_image(img_path, IMG_SIZE)
    image_gpu = image_tensor.unsqueeze(0).to(device)
    image_name = os.path.basename(img_path)

    saliency_map = run_saliency_model(saliency_model, image_tensor, device)
    pred_map = saliency_map.squeeze().detach().cpu().numpy()

    region_masks, class_names, conf_scores = run_yolo_segmentation(
        yolo_model,
        image_tensor,
        conf=YOLO_CONF,
        min_area=MIN_AREA,
        target_hw=IMG_SIZE
    )

    yolo_region_count = len(region_masks)

    region_masks, class_names, conf_scores = merge_wraparound_regions(
        region_masks,
        class_names,
        conf_scores
    )

    added_saliency_count = 0

    # 显著性区域补充消融：disable_supplement=True 时不执行补充
    if (not args.disable_supplement) and len(region_masks) < MIN_YOLO_REGION_COUNT:
        region_masks, class_names, conf_scores, added_saliency_count = add_saliency_supplement_regions(
            region_masks,
            class_names,
            conf_scores,
            pred_map,
            min_area=SUPPLEMENT_MIN_AREA,
            target_count=MIN_YOLO_REGION_COUNT,
            max_add=MAX_SALIENCY_SUPPLEMENT,
            dup_iou_thresh=SALIENCY_DUP_IOU_THRESH
        )

    if yolo_region_count == 0 and added_saliency_count > 0:
        region_type = 'saliency_region'
    elif added_saliency_count > 0:
        region_type = 'yolo11_seg + saliency_region'
    else:
        region_type = 'yolo11_seg'

    if len(region_masks) == 0:
        print('没有找到可用于排序的有效区域')
        return

    ranking_all = rank_regions(
        image_gpu,
        saliency_map,
        region_masks,
        class_names=class_names,
        det_conf=conf_scores
    )

    # NMS 消融：disable_nms=True 时直接取 Top-K，不进行重叠区域去重
    if args.disable_nms:
        important_regions = ranking_all[:TOP_K_OUTPUT]
    else:
        important_regions = select_important_regions(
            ranking_all,
            region_masks,
            top_k=TOP_K_OUTPUT,
            iou_thresh=NMS_IOU_THRESH
        )

    stem = os.path.splitext(image_name)[0]

    # 为消融实验输出文件增加 variant 前缀，避免不同实验组内部覆盖风险
    # 如果使用不同 out_dir，也同样能清楚区分结果。
    saliency_save_path = os.path.join(OUT_DIR, f'{stem}_saliency.png')
    ranking_save_path = os.path.join(OUT_DIR, f'{stem}_important_ranking.png')
    txt_save_path = os.path.join(OUT_DIR, f'{stem}_important_ranking.txt')

    draw_saliency_overlay(image_gpu, saliency_map, saliency_save_path)
    draw_ranking_result(image_gpu, region_masks, important_regions, ranking_save_path)
    save_text_result(image_name, important_regions, region_type, txt_save_path)

    print('\n处理完成')
    print('实验组:', args.variant)
    print('区域来源:', region_type)
    print('YOLO原始候选区域数量:', yolo_region_count)
    print('显著图补充区域数量:', added_saliency_count)
    print('关闭显著性补充:', args.disable_supplement)
    print('关闭Attention:', args.disable_attention)
    print('关闭NMS:', args.disable_nms)
    print('重要性阈值:', IMPORTANCE_THRESHOLD)
    print('Attention温度:', ATTENTION_TEMPERATURE)
    print('Attention融合强度:', ATTENTION_ALPHA)
    print('NMS阈值:', NMS_IOU_THRESH)
    print('检测到候选区域数量:', len(region_masks))
    print('筛选出的重要区域数量:', len(important_regions))
    print('显著性图:', saliency_save_path)
    print('排序结果图:', ranking_save_path)
    print('排序文本:', txt_save_path)

    print('\n重要物体排序结果：')
    for rank_id, item in enumerate(important_regions):
        region_id, score, feat, cls_name, det_score = item
        print(
            f'Rank{rank_id + 1}: Region {region_id:02d}  '
            f'score={score:.4f}  '
            f'class={cls_name}  '
            f'yolo={det_score:.3f}  '
            f'mean={feat["mean_sal"]:.4f}  '
            f'area={feat["area_score"]:.4f}  '
            f'center={feat["center_score"]:.4f}'
        )


if __name__ == '__main__':
    main()