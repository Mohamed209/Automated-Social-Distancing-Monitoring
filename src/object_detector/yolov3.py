import os
import time
import cv2
import numpy as np


class PeopleDetector:
    def __init__(self, yolocfg='yolo_weights/yolov3.cfg',
                 yoloweights='yolo_weights/yolov3.weights',
                 labelpath='yolo_weights/coco.names',
                 confidence=0.5,
                 nmsthreshold=0.4):
        self._yolocfg = yolocfg
        self._yoloweights = yoloweights
        self._confidence = confidence
        self._nmsthreshold = nmsthreshold
        self._labels = open(labelpath).read().strip().split("\n")
        self._colors = np.random.randint(
            0, 255, size=(len(self._labels), 3), dtype="uint8")
        self._net = None
        self._layer_names = None
        self._boxes = []
        self._confidences = []
        self._classIDs = []
        self._centers = []
        self._layerouts = []

    def load_network(self):
        self._net = cv2.dnn.readNetFromDarknet(
            self._yolocfg, self._yoloweights)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._layer_names = [self._net.getLayerNames()[i[0] - 1]
                             for i in self._net.getUnconnectedOutLayers()]
        print("yolov3 loaded successfully\n")

    def predict(self, image):
        #image = cv2.resize(image, (800, 800))
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     [0, 0, 0], 1, crop=False)
        self._net.setInput(blob)
        start = time.time()
        self._layerouts = self._net.forward(self._layer_names)
        end = time.time()
        print("yolo took {:.6f} seconds".format(end - start))
        return(self._layerouts)

    def process_preds(self, image, outs):
        (frameHeight, frameWidth) = image.shape[:2]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self._confidence:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    self._classIDs.append(classId)
                    self._confidences.append(float(confidence))
                    self._boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(
            self._boxes, self._confidences, self._confidence, self._nmsthreshold)
        for i in indices:
            i = i[0]
            box = self._boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.draw_pred(image, self._classIDs[i], self._confidences[i], left,
                           top, left + width, top + height)

    def clear_preds(self):
        self._boxes = []
        self._confidences = []
        self._classIDs = []
        self._centers = []
        self._layerouts = []

    def draw_pred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if self._classIDs:
            assert(classId < len(self._classIDs))
            label = '%s:%s' % (self._labels[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(
            1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        # for out in layerOutputs:
        #     for detection in out:
        #         scores = detection[5:]
        #         classID = np.argmax(scores)
        #         # if classID != 0:  # filter people only
        #         #    continue
        #         confidence = scores[classID]
        #         if confidence > self._confidence:
        #             box = detection[0:4] * np.array([W, H, W, H])
        #             (centerX, centerY, width, height) = box.astype("int")
        #             self._centers.append((centerX, centerY))
        #             x = int(centerX - (width / 2))
        #             y = int(centerY - (height / 2))
        #             self._boxes.append([x, y, int(width), int(height)])
        #             self._confidences.append(float(confidence))
        #             self._classIDs.append(classID)
        #             idxs = cv2.dnn.NMSBoxes(self._boxes, self._confidences, self._confidence,
        #                                     self._threshold)
        #             if len(idxs) > 0:
        #                 # loop over the indexes we are keeping
        #                 for i in idxs.flatten():
        #                     # extract the bounding box coordinates
        #                     (x, y) = (self._boxes[i][0], self._boxes[i][1])
        #                     (w, h) = (self._boxes[i][2], self._boxes[i][3])
        #                     # draw a bounding box rectangle and label on the image
        #                     color = [int(c)
        #                              for c in self._colors[self._classIDs[i]]]
        #                     cv2.rectangle(
        #                         image, (x, y), (x + w, y + h), color, 5)
        #                     text = "{}: {:.4f}".format(
        #                         self._labels[self._classIDs[i]], self._confidences[i])
        #                     cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #                                 1, color, 3)
