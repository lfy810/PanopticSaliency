import torch


def rank_instances(saliency_map, instance_masks):
    """
    saliency_map: [1, 1, H, W]
    instance_masks: list of [H, W] binary masks (torch.Tensor)

    return: list of (instance_id, score), sorted descending
    """

    saliency_map = saliency_map.squeeze()  # [H, W]
    scores = []

    for idx, mask in enumerate(instance_masks):
        mask = mask.float()

        if mask.sum() == 0:
            score = 0.0
        else:
            score = (saliency_map * mask).sum() / mask.sum()

        scores.append((idx, score.item()))

    # 按显著性从大到小排序
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores
