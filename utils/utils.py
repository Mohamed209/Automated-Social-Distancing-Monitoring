import itertools

import numpy as np


def find_min_distance(centers, min_dist=150):
    '''
    return min euclidean distance between predicted anchor boxes
    '''
    comp = list(itertools.combinations(centers, 2))
    critical_distances = {}
    for pts in comp:
        ecdist = np.linalg.norm(np.asarray(pts[0])-np.asarray(pts[1]))
        if ecdist < min_dist:
            critical_distances.update({pts: ecdist})
    severity_index = len(critical_distances)/len(comp)
    return critical_distances, severity_index, comp


def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv
