from object_detector.yolov3 import PeopleDetector
import cv2
net = PeopleDetector()
net.load_network()
image = cv2.imread('test_images/5.jpeg')
net.predict(image, depug=True)
