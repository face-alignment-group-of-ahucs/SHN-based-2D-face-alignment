import cv2
import numpy as np
from utils import *
import matplotlib.pyplot as plt

def get_canonical_shape(keypoints, res):
    dst = np.array([[0, 20], [10, 20], [20, 20]])
    L = keypoints.shape[0]
    if L == 68:
        l, r = 36, 45
    elif L == 98:
        l, r = 60, 72
    else:
        raise ValueError("")
    src = np.array([keypoints[l], keypoints[l] * 0.5 + keypoints[r] * 0.5, keypoints[r]])
    d, z, tform = procrustes(dst, src)
    keypoints = np.dot(keypoints, tform['rotation']) * tform['scale'] + tform['translation']
    gtbox = get_gtbox(keypoints)
    xmin, ymin, xmax, ymax = gtbox
    keypoints -= [xmin, ymin]
    keypoints *= [res / (xmax - xmin), res / (ymax - ymin)]

    return keypoints

def warp(image, src, dst, res, keypoints=None):
    d, Z, meta = procrustes(dst, src)
    M = np.zeros([2, 3], dtype=np.float32)
    M[:2, :2] = meta['rotation'].T * meta['scale']
    M[:, 2] = meta['translation']
    img = cv2.warpAffine(image, M, (res, res))
    if keypoints is not None:
        keypoints = np.dot(keypoints, meta['rotation']) * meta['scale'] + meta['translation']
    return img, keypoints, meta

def crop_from_box(image, box, res, keypoints=None):
    xmin, ymin, xmax, ymax = box
    src = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
    dst = np.array([[0, 0], [0, res - 1], [res - 1, 0], [res - 1, res - 1]])

    return warp(image, src, dst, res, keypoints)

def transform_keypoints(kps, tform, inverse=False):
    if inverse:
        new_kps = np.dot(kps - tform['translation'], np.linalg.inv(tform['rotation'] * tform['scale']))
    else:
        new_kps = np.dot(kps, tform['rotation']) * tform['scale'] + tform['translation']

    return new_kps

def show_preds(image, preds):
    plt.figure()
    plt.imshow(image)
    for pred in preds:
        plt.scatter(pred[:, 0], pred[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()