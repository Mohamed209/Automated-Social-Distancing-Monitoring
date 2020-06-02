from src.object_detector.yolov3 import YoloPeopleDetector
from src.object_detector.postprocessor import PostProcessor
from src.visualization.visualizer import CameraViz
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

# init yolo network , postprocessor and visualization mode
net = YoloPeopleDetector()
net.load_network()

# Process inputs
parser = argparse.ArgumentParser(
    description='Run social distancing meter')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
winName = 'predicted people'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv2.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cv2.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break
    outs = net.predict(frame)
    pp = PostProcessor()
    indices, boxes, ids, confs, centers = pp.process_preds(frame, outs)
    cameraviz = CameraViz(indices, frame, ids, confs, boxes, centers)
    cameraviz.draw_pred()
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # Write the frame with the detection boxes
    if (args.image):
        cv2.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv2.imshow(winName, frame)
