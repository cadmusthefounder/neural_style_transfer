import torch.utils.data
import torchvision.datasets as datasets

from StyleCNN import *
from utils import *

dtype = torch.FloatTensor

# Content and style
style = image_loader("../styles/sun_painting.jpg").type(dtype)
content = image_loader("../contents/peeking_sun.jpg").type(dtype)

pastiche = image_loader("../contents/peeking_sun.jpg").type(dtype)
pastiche.data = torch.randn(input.data.size()).type(dtype)

num_epochs = 31

def main():
    style_cnn = StyleCNN(style, content, pastiche)
    
    for i in range(num_epochs):
        pastiche = style_cnn.train()
    
        if i % 10 == 0:
            print("Iteration: %d" % (i))
            
            path = "../outputs/%d.png" % (i)
            pastiche.data.clamp_(0, 1)
            save_image(pastiche, path)

main()