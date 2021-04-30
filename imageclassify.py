import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable as V


def load_model(path, device):
    model = torchvision.models.vgg19_bn()
    model.Linear = nn.Linear(4096, 2)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    return model


def test(test_img, model):
    img = Image.open(test_img)

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img)
    img = img.unsqueeze(0)
    input = V(img)

    model.eval()
    output = model(input)
    _, preds = torch.max(output, 1)

    return preds


def main():
    # load model
    model_path = 'best_vgg19bn_69.pkl'
    device = torch.device('cpu')

    model = load_model(model_path, device)

    # load image
    img_path = './images/19053.jpg'

    result = test(img_path, model)

    print('this image belongs to class: ([0] for control group and [1] for LPS group)', result.numpy())

if __name__ == '__main__':
    main()
