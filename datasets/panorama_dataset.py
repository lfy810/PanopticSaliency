import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PanoramaDataset(Dataset):
    def __init__(self, img_root, obj_root, img_size=(512, 1024)):
        """
        img_root: data/F-360iSOD/stimulis
        obj_root: data/F-360iSOD/objects
        img_size: (H, W)
        假设：img_root 与 obj_root 下文件名一一对应（如 001.png）
        """
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

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]               # 例如 001.png
        img_path = os.path.join(self.img_root, name)
        obj_path = os.path.join(self.obj_root, name)

        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)

        # object mask 作为灰度读入（不转 tensor；后续用 numpy 处理连通域更方便）
        obj_mask = Image.open(obj_path).convert('L')
        obj_mask = obj_mask.resize((self.img_size[1], self.img_size[0]), resample=Image.NEAREST)

        return image, obj_mask, name
