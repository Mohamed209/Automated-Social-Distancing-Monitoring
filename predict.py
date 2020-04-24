from scipy.spatial import distance
from model.model import CSRNet
from torchvision import datasets, transforms
from matplotlib import cm as c
from sklearn.cluster import KMeans
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
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
    'test-data/images/4.jpg').convert('RGB'))
output = model(img.unsqueeze(0))

count = int(output.detach().cpu().sum().numpy())
print("Predicted Count : ", count)
temp = np.asarray(output.detach().cpu().reshape(
    output.detach().cpu().shape[2], output.detach().cpu().shape[3]), dtype=np.float)*255
print("density matrix shape \n", temp.shape)
# temp = cv2.GaussianBlur(temp, (3, 3), 0)
plt.imshow(temp, cmap=c.jet)
plt.colorbar()
plt.show()
# cluster points
#kmeans = KMeans(n_clusters=count, n_jobs=-1)
#kmeans.fit(temp)
#print(kmeans.cluster_centers_.shape)
dist = distance.cdist(temp, temp)
np.fill_diagonal(dist, np.nan)
print("min dist \n", np.nanmin(dist))
