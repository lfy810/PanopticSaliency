import os
import torch
import cv2
import numpy as np

from datasets.panorama_train_dataset import PanoramaTrainDataset
from models.full_model import DistortionAwareSaliencyModel


# ============ 配置区 ============
IMG_ROOT = 'data/F-360iSOD/stimulis'
MASK_ROOT = 'data/F-360iSOD/objects'
IMG_SIZE = (512, 1024)

MODEL_PATH = 'checkpoints/best_model.pth'
OUT_DIR = 'outputs/trained_pred'
N_SAMPLES = 10
# ===============================


def save_pred_map(image_tensor, pred_map, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pred = pred_map.squeeze().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

    vis = cv2.addWeighted(img, 0.6, pred, 0.4, 0)

    save_path = os.path.join(save_dir, name.replace('.png', '_pred.png'))
    cv2.imwrite(save_path, vis)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('使用设备:', device)

    dataset = PanoramaTrainDataset(
        img_root=IMG_ROOT,
        mask_root=MASK_ROOT,
        img_size=IMG_SIZE
    )

    model = DistortionAwareSaliencyModel(img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    n = min(N_SAMPLES, len(dataset))
    print('本次推理:', n)

    with torch.no_grad():
        for i in range(n):
            image, mask, name = dataset[i]
            image_gpu = image.unsqueeze(0).to(device)

            logits = model(image_gpu)
            pred = torch.sigmoid(logits)

            save_pred_map(image, pred, name, OUT_DIR)
            print(f'[{i+1:03d}/{n:03d}] {name}  saved')

    print('\n结果已保存到:', OUT_DIR)


if __name__ == '__main__':
    main()