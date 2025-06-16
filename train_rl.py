import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count(), torch.cuda.get_device_name(0))
