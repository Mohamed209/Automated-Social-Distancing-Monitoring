import torch
import matplotlib.pyplot as plt
import numpy as np
from model.model import CSRNet
from torchvision import datasets, transforms
from matplotlib import cm as c
from PIL import Image
model = CSRNet()
# loading the trained weights
checkpoint = torch.load(
    'pretrained-weights/0model_best.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])
# check preds for single image
img = transform(Image.open(
    'test-data/images/960x0.jpg').convert('RGB'))
output = model(img.unsqueeze(0))
print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(
    output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
print("density matrix \n", temp)
plt.imshow(temp, cmap=c.jet)
plt.show()
# TODO : discretize the density map
