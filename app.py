import dash
import dash_core_components as dcc
import dash_html_components as html
import time
from flask import Flask, Response
from src.object_detector.yolov3 import YoloPeopleDetector
from src.object_detector.postprocessor import PostProcessor
from src.visualization.visualizer import CameraViz
import cv2


def stream_test_local_video(path):
    cap = cv2.VideoCapture(path)
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # outs = net.predict(frame)
            # pp = PostProcessor()
            # indices, boxes, ids, confs, centers = pp.process_preds(frame, outs)
            # cameraviz = CameraViz(indices, frame, ids, confs, boxes, centers)
            # cameraviz.draw_pred()
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break


server = Flask(__name__)
app = dash.Dash(__name__, server=server)


@server.route('/video_feed')
def video_feed():
    return Response(stream_test_local_video(path='data/test_videos/cut3.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div([
    html.H1("Real Time Social Distancing Monitor", id='page_header',),
    html.Div(html.Img(src="/video_feed", height=500, width=800))
])

if __name__ == '__main__':
    # init yolo network , postprocessor and visualization mode
    # net = YoloPeopleDetector()
    # net.load_network()
    app.run_server(debug=True)
