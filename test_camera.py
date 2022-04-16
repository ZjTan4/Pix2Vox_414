import torch

num_views = 4
batch_size = 32

elev = torch.linspace(0, 0, num_views * batch_size)
azim = torch.linspace(-180, 180, num_views) + 180.0
azim = azim.expand(batch_size, num_views).T.flatten()
print(azim[64])
