from image import *
from content_loss import ContentLoss
from style_loss import StyleLoss
from normalization import Normalization

import os
import copy
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

params = {
    'content_image': 'charlton6.jpg',
    'style_image': 'picasso.jpg',
    'output_image': 'output18_{}.jpg'
}

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, output_image_path, 
                       content_layers, style_layers, device,
                       num_steps=300, style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean, 
        normalization_std, 
        style_img, 
        content_img,
        content_layers,
        style_layers,
        device)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                save_image(input_img, output_image_path.format(run[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers, style_layers, device):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (1028, 1028) if torch.cuda.is_available() else (512, 512)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    content_image_path = os.path.join(current_directory, '..', 'contents', params['content_image'])
    style_image_path = os.path.join(current_directory, '..', 'styles', params['style_image'])
    output_image_path = os.path.join(current_directory, '..', 'outputs', params['output_image'])

    content_image = image_loader(content_image_path, image_size, device=device)
    style_image = image_loader(style_image_path, image_size, device=device)
    input_image = content_image.clone()
    print(content_image.size())
    print(style_image.size())
    assert content_image.size() == style_image.size()

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    run_style_transfer(
        cnn, 
        cnn_normalization_mean, 
        cnn_normalization_std,
        content_image, 
        style_image, 
        input_image,
        output_image_path,
        content_layers_default,
        style_layers_default,
        device,
        num_steps=300)
    
if __name__ == "__main__":
    main()