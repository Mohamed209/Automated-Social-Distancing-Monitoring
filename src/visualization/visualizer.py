from utils.utils import find_min_distance
import cv2
import numpy as np
#from math import round


class Visualizer:
    def __init__(self, critical_line_color=(0, 0, 255), critical_line_thickness=5):
        self.critical_line_color = critical_line_color
        self.critical_line_thickness = critical_line_thickness

    def draw_pred(self):
        pass


class CameraViz(Visualizer):
    def __init__(self, nmsboxes, frame, classIds, confs, boxes, centers, labelpath='yolo_weights/coco.names',
                 detected_object_rect_color=(255, 178, 50), detected_object_rect_thickness=3,
                 label_font=cv2.FONT_HERSHEY_SIMPLEX, label_fontscale=0.5, label_font_thickness=1,
                 label_rect_color=(255, 255, 255), label_text_color=(0, 0, 0), meter_fontscale=1, meter_font_thickness=2,
                 meter_text_color=(255, 0, 0)):
        super().__init__()
        self._labelpath = labelpath
        self._labels = open(self._labelpath).read().strip().split("\n")
        self.detected_object_rect_color = detected_object_rect_color
        self.detected_object_rect_thickness = detected_object_rect_thickness
        self.label_font = label_font
        self.label_fontscale = label_fontscale
        self.label_font_thickness = label_font_thickness
        self.label_rect_color = label_rect_color
        self.label_text_color = label_text_color
        self.meter_fontscale = meter_fontscale
        self.meter_font_thickness = meter_font_thickness
        self.meter_text_color = meter_text_color
        self.__nmsboxes = nmsboxes
        self.__frame = frame
        self.__boxes = boxes
        self.__classIds = classIds
        self.__confs = confs
        self.__centers = centers
        self.critical_dists = {}
        self.alldists = []
        self.sev_idx = 0.0

    def draw_pred(self):
        # TODO : more modularization of draw_pred() functions
        for i in self.__nmsboxes:
            i = i[0]
            box = self.__boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(self.__frame, (left, top),
                          (left+width, top+height), self.detected_object_rect_color, self.detected_object_rect_thickness)
            label = '%.2f' % self.__confs[i]
            label = '%s:%s' % (self._labels[self.__classIds[i]], label)
            labelSize, baseLine = cv2.getTextSize(
                label, self.label_font, self.label_fontscale, self.label_font_thickness)
            top = max(top, labelSize[1])
            cv2.rectangle(self.__frame, (left, top - round(1.5*labelSize[1])), (left + round(
                1.5*labelSize[0]), top + baseLine), self.label_rect_color, cv2.FILLED)
            cv2.putText(self.__frame, label, (left, top),
                        self.label_font, self.label_fontscale, self.label_text_color, self.label_font_thickness)

            self.critical_dists, self.sev_idx, self.alldists = find_min_distance(
                self.__centers)
            # TODO : move common attributes between camera and birds eye view to base class
            for dist in self.critical_dists:
                cv2.line(self.__frame, dist[0], dist[1],
                         self.critical_line_color, self.critical_line_thickness)
            # show severity index
            self.sev_idx = round(self.sev_idx, 3)*100
            cv2.putText(self.__frame, "Severity Index : "+str(self.sev_idx)+' %', (50, 50),
                        self.label_font, self.meter_fontscale, self.meter_text_color, self.meter_font_thickness)


class BirdseyeViewTransformer:
    '''
    morphs the perspective view into a bird’s-eye (top-down) view
    note : four_pts needs to be manually calibrated
    '''

    def __init__(self, frame, four_pts=np.float32(
            [[230, 730], [950, 950], [1175, 175], [1570, 230]]), scale_w=1.2/2, scale_h=4/2):
        self.__four_pts = four_pts
        self.__scale_w = scale_w
        self.__scale_h = scale_h
        self.__dst = np.float32(
            [[0, frame.shape[1]], [frame.shape[0], frame.shape[1]], [0, 0], [frame.shape[0], 0]])

    def map_point_birdsview(self, point):
        M = cv2.getPerspectiveTransform(self.__dst, self.__four_pts)
        warped_pt = cv2.perspectiveTransform(point, M)[0][0]
        warped_pt_scaled = [int(warped_pt[0] * self.__scale_w),
                            int(warped_pt[1] * self.__scale_h)]
        return warped_pt_scaled


class BirdseyeViewViz(Visualizer):
    def __init__(self, node_radius=10, color_node=(0, 255, 0), thickness_node=20, cents=None):
        super().__init__()
        self.node_radius = node_radius
        self.color_node = color_node
        self.thickness_node = thickness_node
        self.__cents = cents

    def draw_pred(self):
        pass
