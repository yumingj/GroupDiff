# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand



class OpenposeDetector:
    def __init__(self, device=None):
        body_modelpath = './pretrained_models/body_pose_model.pth'
        hand_modelpath = './pretrained_models/hand_pose_model.pth'

        self.body_estimation = Body(body_modelpath, device)
        self.hand_estimation = Hand(hand_modelpath, device)

    def __call__(self, oriImg, hand=False):
        oriImg = oriImg[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            canvas = np.zeros_like(oriImg)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            if hand:
                hands_list = util.handDetect(candidate, subset, oriImg)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
                all_hand_peaks = [peak.tolist() for peak in all_hand_peaks]
                return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist(), all_hand_peaks = all_hand_peaks)
            else:
                return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())
