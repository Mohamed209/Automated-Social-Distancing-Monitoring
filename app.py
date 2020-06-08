import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import time
import cv2
import dash_bootstrap_components as dbc
import datetime
import numpy as np
from flask import Flask, Response
from src.object_detector.yolov3 import YoloPeopleDetector
from src.object_detector.postprocessor import PostProcessor
from src.visualization.visualizer import CameraViz
from src.data_feed.data_feeder import ViolationsFeed
from dash.dependencies import Input, Output, State


def stream_test_local_video(path):
    cap = cv2.VideoCapture(path)
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            outs = net.predict(frame)
            pp = PostProcessor()
            indices, boxes, ids, confs, centers = pp.process_preds(frame, outs)
            cameraviz = CameraViz(indices, frame, ids, confs, boxes, centers)
            cameraviz.draw_pred()
            # feed critical dists and non critical into viofeed
            vf.feed_new((cameraviz.critical_dists, cameraviz.alldists))
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break


server = Flask(__name__)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[dbc.themes.SUPERHERO])


@server.route('/video_feed')
def video_feed():
    return Response(stream_test_local_video(path='data/test_videos/cut3.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div([dbc.Container(
    [
        html.H1("Real Time Social Distancing Monitor",
                style={'text-align': 'center'}),
        dbc.Row(
            [
                dbc.Col(html.Div(id='video_stream', children=[html.Img(
                    src="/video_feed", height=400, width=800)]), md=4.5),
                dbc.Col(dcc.Graph(id="violations"))
            ],
            align="start"
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="phones-graphh")),
                dbc.Col(dcc.Graph(id="phones-graphhh"))
            ],
            align="start",
        )
    ],
    fluid=True,
),
    dcc.Interval(
    id='interval-component',
    interval=5*1000,  # update graph every n*1000 millisecond
    n_intervals=0
)]
)


@app.callback(Output('violations', 'figure'), [Input('interval-component', 'n_intervals')])
def update_violations_graph(n):
    # data collection
    vio, nonvio = vf.get_feed()
    t = datetime.datetime.now()
    fig.add_trace(go.Bar(name='Violations', x=[t], y=vio))
    fig.add_trace(go.Bar(name='Non Violations', x=[t], y=nonvio))
    # fig = go.Figure(data=[
    #     go.Bar(name='Violations', x=[t], y=vio),
    #     go.Bar(name='Non Violations', x=[t], y=nonvio)
    # ])
    vf.clear_feed()
    return fig


if __name__ == '__main__':
    # init data feed pipes
    vf = ViolationsFeed()
    fig = go.Figure()
    fig.update_layout(
        barmode='stack', title_text='Violations VS Non Violations Graph')
    # init and load yolo network
    net = YoloPeopleDetector()
    net.load_network()
    app.run_server(debug=True)
