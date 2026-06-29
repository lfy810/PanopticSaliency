import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PanoramaTrainDataset(Dataset):
    def __init__(self, img_root, obj_root, img_size=(512, 1024)):
        self.img_root = img_root
        self.obj_root = obj_root
        self.img_size = img_size

        self.image_list = sorted([
            f for f in os.listdir(img_root)
            if f.lower().endswith('.png')
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]

        img_path = os.path.join(self.img_root, name)
        mask_path = os.path.join(self.obj_root, name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # 二值化
        mask = (mask > 0).float()

        return image, mask, name