import os
import random
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms

from models.full_model import DistortionAwareSaliencyModel


# ============ 配置区 ============
F360_IMG_ROOT = 'data/F-360iSOD/stimulis'
F360_MASK_ROOT = 'data/F-360iSOD/objects'

IMG_SIZE = (512, 1024)

BATCH_SIZE = 1
NUM_WORKERS = 0
SEED = 42
VAL_RATIO = 0.2

# 从当前最优模型继续微调
OLD_MODEL_PATH = 'checkpoints/best_model_sgdaf_v7.pth'

SAVE_DIR = 'checkpoints'
FINAL_SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_sgdaf_v8.pth')

# 冲刺训练参数
FINETUNE_EPOCHS = 20
FINETUNE_LR = 5e-6
MIN_LR = 1e-6
GRAD_CLIP = 1.0

# 跳过全黑 / 太小 mask
MIN_MASK_AREA = 50
# ===============================


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SaliencyDataset(Dataset):
    def __init__(self, img_root, mask_root, img_size=(512, 1024), min_mask_area=50):
        self.img_root = img_root
        self.mask_root = mask_root
        self.img_size = img_size
        self.min_mask_area = min_mask_area
        self.to_tensor = transforms.ToTensor()

        raw_images = sorted([
            f for f in os.listdir(img_root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        self.image_list = []
        self.skipped_list = []

        for name in raw_images:
            mask_path = self._find_mask_path_safe(name)
            if mask_path is None:
                self.skipped_list.append((name, 'mask_missing'))
                continue

            try:
                mask = Image.open(mask_path).convert('L')
                mask_np = torch.from_numpy(
                    __import__('numpy').array(mask)
                )
                area = int((mask_np > 0).sum().item())

                if area < self.min_mask_area:
                    self.skipped_list.append((name, f'mask_area={area}'))
                    continue

                self.image_list.append(name)

            except Exception as e:
                self.skipped_list.append((name, f'error={str(e)}'))

        print('\n========== 数据集清洗结果 ==========')
        print('原始样本数:', len(raw_images))
        print('有效样本数:', len(self.image_list))
        print('跳过样本数:', len(self.skipped_list))
        if len(self.skipped_list) > 0:
            print('跳过样本示例:')
            for item in self.skipped_list[:20]:
                print(' ', item)
        print('====================================\n')

    def __len__(self):
        return len(self.image_list)

    def _find_mask_path_safe(self, name):
        mask_path = os.path.join(self.mask_root, name)
        if os.path.exists(mask_path):
            return mask_path

        stem = os.path.splitext(name)[0]
        candidates = [
            stem + '.png',
            stem + '.jpg',
            stem + '.jpeg',
            stem + '.bmp'
        ]

        for c in candidates:
            p = os.path.join(self.mask_root, c)
            if os.path.exists(p):
                return p

        return None

    def _find_mask_path(self, name):
        p = self._find_mask_path_safe(name)
        if p is None:
            raise FileNotFoundError(f'找不到对应 mask: {name}')
        return p

    def __getitem__(self, idx):
        name = self.image_list[idx]

        img_path = os.path.join(self.img_root, name)
        mask_path = self._find_mask_path(name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        mask = (mask > 0).float()

        return image, mask, name


def split_dataset(dataset, val_ratio=0.2, seed=42):
    total = len(dataset)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    val_size = max(1, int(total * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


def dice_loss(logits, target, smooth=1.0):
    pred = torch.sigmoid(logits)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
    return loss.mean()


def load_weights(model, weight_path, device):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f'找不到旧模型权重: {weight_path}')

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=True)

    print('\n已加载旧模型权重:', weight_path)
    return model


def train_one_epoch(model, loader, optimizer, bce_loss_fn, device):
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        bce = bce_loss_fn(logits, masks)
        d_loss = dice_loss(logits, masks)

        loss = bce + d_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    return total_loss / max(valid_batches, 1)


def validate(model, loader, bce_loss_fn, device):
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            bce = bce_loss_fn(logits, masks)
            d_loss = dice_loss(logits, masks)

            loss = bce + d_loss

            total_loss += loss.item()
            valid_batches += 1

    return total_loss / max(valid_batches, 1)


def main():
    set_seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('使用设备:', device)

    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = SaliencyDataset(
        img_root=F360_IMG_ROOT,
        mask_root=F360_MASK_ROOT,
        img_size=IMG_SIZE,
        min_mask_area=MIN_MASK_AREA
    )

    train_set, val_set = split_dataset(dataset, val_ratio=VAL_RATIO, seed=SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print('\n===== SG-DAF v5 冲刺微调 =====')
    print('总有效样本数:', len(dataset))
    print('训练集:', len(train_set))
    print('验证集:', len(val_set))
    print('旧模型:', OLD_MODEL_PATH)
    print('保存路径:', FINAL_SAVE_PATH)
    print('初始学习率:', FINETUNE_LR)
    print('最低学习率:', MIN_LR)
    print('训练轮数:', FINETUNE_EPOCHS)

    model = DistortionAwareSaliencyModel(img_size=IMG_SIZE).to(device)
    model = load_weights(model, OLD_MODEL_PATH, device)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FINETUNE_EPOCHS,
        eta_min=MIN_LR
    )

    best_val_loss = float('inf')

    for epoch in range(FINETUNE_EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss_fn, device)
        val_loss = validate(model, val_loader, bce_loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), FINAL_SAVE_PATH)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        t_cost = time.time() - t0

        print(
            f'Epoch [{epoch + 1}/{FINETUNE_EPOCHS}]  '
            f'train_loss={train_loss:.4f}  '
            f'val_loss={val_loss:.4f}  '
            f'lr={current_lr:.8f}  '
            f'time={t_cost:.1f}s'
        )

    print('\nSG-DAF 冲刺微调完成')
    print('最佳模型已保存到:', FINAL_SAVE_PATH)
    print('best_val_loss:', round(best_val_loss, 4))


if __name__ == '__main__':
    main()