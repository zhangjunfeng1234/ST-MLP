import torch
from mlp_mixer_pytorch import MLPMixer
model = MLPMixer(
    image_size = (19, 128),
    channels = 2,
    patch_size1 = 19,
    patch_size2 = 1,
    dim = 608,
    depth = 12,
    num_classes = 1000
)
print(model)
img = torch.randn(32, 2, 19, 128)
pred = model(img) # (1, 1000)
print(pred)