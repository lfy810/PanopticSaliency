import torch
print("torch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
