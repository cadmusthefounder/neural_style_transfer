from PIL import Image

import torch
import torchvision.transforms as transforms

def image_loader(image_name, image_size, device=torch.device("cpu")):
    loader = transforms.Compose([
        transforms.Resize(image_size), # resize imported image
        transforms.ToTensor() # transform it into a torch tensor
    ])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def image_unloader(image):
    return transforms.ToPILImage(image)  

def save_image(image, image_path):
    torchvision.utils.save_image(image, image_path)
