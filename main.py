from object_detector.yolov3 import PeopleDetector
from scipy.spatial import distance
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
net = PeopleDetector()
net.load_network()
image = cv2.imread('test_images/g2.jpg')
cents = net.predict(image, depug=True)
print(cents)
dist = distance.cdist(cents, cents)
comp = list(itertools.combinations(cents, 2))
print(comp)
np.fill_diagonal(dist, np.nan)
print("min dist \n", np.nanmin(dist))
cents = np.array(cents).reshape(-1, 2)
plt.scatter(x=cents[:, 0], y=cents[:, 1])
plt.show()
for i in range(len(comp)):
    plt.plot([comp[i][0][0], comp[i][1][0]], [
             comp[i][0][1], comp[i][1][1]], '-o')
plt.title('2D map of predicted people')
plt.show()
# cap = cv2.VideoCapture('chaplin.mp4')
# if (cap.isOpened() == False):
#     print("Error opening video stream or file")
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         net.predict(frame, depug=True)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()
