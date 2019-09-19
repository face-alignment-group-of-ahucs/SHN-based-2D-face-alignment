import cv2
import dlib
import torch
import numpy as np
from model import FAN
from transform import *
from utils import *
import matplotlib.pyplot as plt

#1. initialize model and weights
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint = 'best_checkpoint.pth.tar' # checkpoint path
model = FAN(3,68)
state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

#2. load image and perform dlib face detection and dlib face alignment
predictons = []

image = cv2.imread('obama.jpg')[:,:,::-1]
detector = dlib.get_frontal_face_detector()
# download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
dets = detector(image, 1)
for idx, det in enumerate(dets):
    shape = landmark_predictor(image, det)
    dlib_shape = []
    for i in range(68):
        dlib_shape.append([shape.part(i).x, shape.part(i).y])
    dlib_shape = np.array(dlib_shape)
    can_shape = get_canonical_shape(dlib_shape, 112)
    can_shape += [8, 8]
    img, _, meta = warp(image, dlib_shape, can_shape, 128)
    # convert to tensor
    img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255)
    img.unsqueeze_(0)
    img = img.to(device)
    with torch.no_grad():
        out = model(img)
        out = get_preds(out)
        if out.is_cuda:
            out = out.cpu()
        pred = out.squeeze(0).numpy()
    pred = transform_keypoints(pred, meta, inverse=True)
    predictons.append(pred)

# 3. plot landmarks
show_preds(image, predictons)
