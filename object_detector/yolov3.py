import os
import time
import cv2
import numpy as np


class PeopleDetector:
    def __init__(self, yolocfg='yolo-coco/yolov3-spp.cfg',
                 yoloweights='yolo-coco/yolov3-spp.weights',
                 labelpath='yolo-coco/coco.names',
                 confidence=0.5,
                 threshold=0.3):
        self._yolocfg = yolocfg
        self._yoloweights = yoloweights
        self._confidence = confidence
        self._threshold = threshold  # NMS
        self._labels = open(labelpath).read().strip().split("\n")
        self._colors = np.random.randint(
            0, 255, size=(len(self._labels), 3), dtype="uint8")
        self._net = None
        self._layer_names = None
        self._boxes = []
        self._confidences = []
        self._classIDs = []
        self._centers = []

    def load_network(self):
        print("loading yolov3 network\n")
        self._net = cv2.dnn.readNetFromDarknet(
            self._yolocfg, self._yoloweights)
        self._layer_names = [self._net.getLayerNames()[i[0] - 1]
                             for i in self._net.getUnconnectedOutLayers()]
        print("yolov3 loaded successfully\n")

    def predict(self, image, depug=False):
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self._net.setInput(blob)
        start = time.time()
        layerOutputs = self._net.forward(self._layer_names)
        for out in layerOutputs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self._confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    self._centers.append((centerX, centerY))
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    self._boxes.append([x, y, int(width), int(height)])
                    self._confidences.append(float(confidence))
                    self._classIDs.append(classID)
                    idxs = cv2.dnn.NMSBoxes(self._boxes, self._confidences, self._confidence,
                                            self._threshold)
                    if len(idxs) > 0:
                        # loop over the indexes we are keeping
                        for i in idxs.flatten():
                            # extract the bounding box coordinates
                            (x, y) = (self._boxes[i][0], self._boxes[i][1])
                            (w, h) = (self._boxes[i][2], self._boxes[i][3])
                            # draw a bounding box rectangle and label on the image
                            color = [int(c)
                                     for c in self._colors[self._classIDs[i]]]
                            cv2.rectangle(
                                image, (x, y), (x + w, y + h), color, 5)
                            text = "{}: {:.4f}".format(
                                self._labels[self._classIDs[i]], self._confidences[i])
                            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, color, 3)

            end = time.time()
            print("yolo took {:.6f} seconds".format(end - start))
            if depug:
                PeopleDetector.visualize_preds(image)
            return self._centers

    @staticmethod
    def visualize_preds(image):
        cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)
        cv2.imshow('prediction', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
