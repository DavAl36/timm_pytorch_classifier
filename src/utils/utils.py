import yaml

# Return dictionary from yaml file

def load_config_yml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Return all target value from a test_loader

def extract_targets(test_dataloader):
    all_targets = []
    for batch in test_dataloader:
        _, targets = batch
        all_targets.extend(targets.tolist())

    return all_targets




# Converts the N test_loader elements into images
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

def tensors_to_img(test_dataloader, N):

    primo_batch = next(iter(test_dataloader))
    dimensioni_primo_batch = primo_batch[0].size()
    primo_batch = primo_batch[0]     
    # Pytorch Tensor [batch_size, canali, altezza, larghezza]
    tensor = primo_batch
    # Normalize between 0-1
    tensor = torch.clamp(tensor, 0, 1)

    # Convert tensor to PIL image
    def tensor_to_image(tensor):
        tensor = tensor.mul(255).byte()
        image = TF.to_pil_image(tensor)
        return image
    if tensor.size(0) >= N:
        print(f"Print the first {N} elements of test_dataloader") 
        # Show images
        for i in range(N):
            img = tensor_to_image(tensor[i])
            img.show()
    else:
        print(f"Cannot print images because N:{N} is greater than tensor size:{tensor.size(0)}") 
