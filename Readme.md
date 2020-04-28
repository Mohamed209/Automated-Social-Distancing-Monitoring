## Open Source Implementation of landing AI Social Distancing Detector

" In the fight against the coronavirus, social distancing has proven to be a very effective measure to slow down the spread of the disease. While millions of people are staying at home to help flatten the curve, many of our customers in the manufacturing and pharmaceutical industries are still having to go to work everyday to make sure our basic needs are met.

To complement our customers’ efforts and to help ensure social distancing protocol in their workplace, Landing AI has developed an AI-enabled social distancing detection tool that can detect if people are keeping a safe distance from each other by analyzing real time video streams from the camera.

For example, at a factory that produces protective equipment, technicians could integrate this software into their security camera systems to monitor the working environment with easy calibration steps. As the demo shows below, the detector could highlight people whose distance is below the minimum acceptable distance in red, and draw a line between to emphasize this. The system will also be able to issue an alert to remind people to keep a safe distance if the protocol is violated "

source : https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/ 

`check original results` 

![original results](demo.gif)

this repo acts as open source python implementation to the demo porposed by landingAI

## System Architecture

![system arch](system_arch.png)

## How to use the repo

### 1.clone the repo

### 2. Install dependencies (reccommended in virtual environment)

``` 
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

###  3.download YOLOV3 pretrained weights and architecture 

``` 
mkdir yolo_weights
cd yolo_weights
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

### 4.run the code

``` 
python main.py --video 'video path'
```

## check sample result of the implementation

![my results](mydemo.gif)

## Citation

test video taken from http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi

``` 
@article{yolov3, 
  title={YOLOv3: An Incremental Improvement}, 
  author={Redmon, Joseph and Farhadi, Ali}, 
  journal = {arXiv}, 
  year={2018}
}
```

