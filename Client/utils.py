import os
import sys
import time
import numpy as np
import copy
import mtcnn
from mtcnn import MTCNN
import cv2
import requests
import json


print("mtcnn_version: {}".format(mtcnn.__version__))
print("cv2_version: {}".format(cv2.__version__))


class ImgSend:
    def __init__(self, host, port, debug_mode=False):
        self.host = host
        self.port = port
        self.url_recognize = host + ":" + str(port) + "/face-recognize"
        self.url_create = host + ":" + str(port) + "/create"
        self.dbg_mode = debug_mode
    
    def send_image_recognize(self, img):
        _, img_encoded = cv2.imencode(".png", img)

        data = img_encoded.tostring()
        headers = {"content-type": "image/png"}
        if self.dbg_mode:
            print("Sending request... ")
            #print(data)
        t1 = time.time()
        response = requests.post(self.url_recognize, data=data, headers=headers)
        t2 = time.time()
        dt = t2-t1
        if self.dbg_mode:
            print("Request processed: " + str(dt) + " sec")
        
        result = json.loads(response.text)
        return result

    def send_image_create(self, name, img):
        _, img_encoded = cv2.imencode(".png", img)
        headers = {"Content-Type": "application/json"}
        data = {
            "name": name,
            "image": img_encoded.tostring().decode('ISO-8859-1')
        }
        if self.dbg_mode:
            print("Sending request... ")
        t1 = time.time()
        response = requests.post(self.url_create, data=json.dumps(data), headers=headers)
        t2 = time.time()
        dt = t2 - t1
        if self.dbg_mode:
            print("Request processed: " + str(dt) + " sec")

        result = response.json()
        return result


class MTCNNDetector:
    def __init__(self, min_size, min_confidence):
        self.min_size = min_size
        self.f_detector = MTCNN(min_face_size=min_size)
        self.min_confidence = min_confidence
    
    def detect(self, frame):
        faces = self.f_detector.detect_faces(frame)
        
        detected = []
        for (i, face) in enumerate(faces):
            f_conf = face['confidence']
            if f_conf >= self.min_confidence:
                detected.append(face)
        
        return detected
    
    def extract(self, frame, face):
        (x1, y1, w, h) = face['box']
        (l_eye, r_eye, nose, mouth_l, mouth_r) = Utils.get_keypoints(face)
        
        f_cropped = copy.deepcopy(face)
        move = (-x1, -y1)
        l_eye = Utils.move_point(l_eye, move)
        r_eye = Utils.move_point(r_eye, move)
        nose = Utils.move_point(nose, move)
        mouth_l = Utils.move_point(mouth_l, move)
        mouth_r = Utils.move_point(mouth_r, move)
            
        f_cropped['box'] = (0, 0, w, h)
        f_img = frame[y1:y1+h, x1:x1+w].copy()
            
        f_cropped = Utils.set_keypoints(f_cropped, (l_eye, r_eye, nose, mouth_l, mouth_r))
        
        return f_cropped, f_img


class Utils:    
    @staticmethod
    def draw_face(face, color, frame, draw_points=True, draw_rect=True, n_data=None):
        (x1, y1, w, h) =  face['box']
        confidence = face['confidence']
        x2 = x1+w
        y2 = y1+h
        if draw_rect:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        y3 = y1-12
        if not (n_data is None):
            (name, conf) = n_data
            text = name+ (" %.3f" % conf)
        else:
            text = "%.3f" % confidence
        
        cv2.putText(frame, text, (x1, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        if draw_points:
            (l_eye, r_eye, nose, mouth_l, mouth_r) = Utils.get_keypoints(face)
            Utils.draw_point(l_eye, color, frame)
            Utils.draw_point(r_eye, color, frame)
            Utils.draw_point(nose, color, frame)
            Utils.draw_point(mouth_l, color, frame)
            Utils.draw_point(mouth_r, color, frame)
        
    @staticmethod
    def get_keypoints(face):
        keypoints = face['keypoints']
        l_eye = keypoints['left_eye']
        r_eye = keypoints['right_eye']
        nose = keypoints['nose']
        mouth_l = keypoints['mouth_left']
        mouth_r = keypoints['mouth_right']
        return (l_eye, r_eye, nose, mouth_l, mouth_r)
    
    def set_keypoints(face, points):
        (l_eye, r_eye, nose, mouth_l, mouth_r) = points
        keypoints = face['keypoints']
        keypoints['left_eye'] = l_eye
        keypoints['right_eye'] = r_eye
        keypoints['nose'] = nose
        keypoints['mouth_left'] = mouth_l
        keypoints['mouth_right'] = mouth_r
        
        return face
        
    @staticmethod
    def move_point(point, move):
        (x, y) = point
        (dx, dy) = move
        res = (x+dx, y+dy)
        return res
        
    @staticmethod
    def draw_point(point, color, frame):
        (x, y) =  point
        x1 = x-1
        y1 = y-1
        x2 = x+1
        y2 = y+1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
    @staticmethod
    def draw_faces(faces, color, frame, draw_points=True, draw_rect=True, names=None):
        for (i, face) in enumerate(faces):
            n_data = None
            if not (names is None):
                n_data = names[i]
            Utils.draw_face(face, color, frame, draw_points, draw_rect, n_data)


class FileUtils:
    @staticmethod
    def get_files(folder):
        files = []
        filenames = os.listdir(folder)
        for (i, f_name) in enumerate(filenames):
            full_path = os.path.join(folder, f_name)
            files.append(full_path)
        return files


class FaceAlign:
    def __init__(self, size):
        self.size = size
    
    def align_point(self, point, M):
        (x, y) = point
        p = np.float32([[[x, y]]])
        p = cv2.perspectiveTransform(p, M)
        
        return int(p[0][0][0]), int(p[0][0][1])

    def align(self, frame, face):
        (x1, y1, w, h) = face['box']
        (l_eye, r_eye, nose, mouth_l, mouth_r) = Utils.get_keypoints(face)
        
        pts1, pts2 = self.get_perspective_points(l_eye, r_eye, nose, mouth_l, mouth_r)
        s = self.size
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(frame, M, (s, s))
        
        f_aligned = copy.deepcopy(face)
        f_aligned['box'] = (0, 0, s, s)
        f_img = dst
        
        l_eye = self.align_point(l_eye, M)
        r_eye = self.align_point(r_eye, M)
        nose = self.align_point(nose, M)
        mouth_l = self.align_point(mouth_l, M)
        mouth_r = self.align_point(mouth_r, M)
        
        f_aligned = Utils.set_keypoints(f_aligned, (l_eye, r_eye, nose, mouth_l, mouth_r))
        
        return f_aligned, f_img


class FaceAlignNose(FaceAlign):
    def get_perspective_points(self, l_eye, r_eye, nose, mouth_l, mouth_r):
        (xl, yl) = l_eye
        (xr, yr) = r_eye
        (xn, yn) = nose
        (xm, ym) = ( 0.5*(xl+xr), 0.5*(yl+yr) )
        (dx, dy) = (xn-xm, yn-ym)
        (xl2, yl2) = (xl+2.0*dx, yl+2.0*dy)
        (xr2, yr2) = (xr+2.0*dx, yr+2.0*dy)
        
        s = self.size
        pts1 = np.float32([[xl, yl], [xr, yr], [xr2, yr2], [xl2, yl2]])
        pts2 = np.float32([[s*0.25, s*0.25], [s*0.75, s*0.25], [s*0.75, s*0.75], [s*0.25,s*0.75]])
        
        return pts1, pts2


class FaceAlignMouth(FaceAlign):
    def get_perspective_points(self, l_eye, r_eye, nose, mouth_l, mouth_r):
        (xl, yl) = l_eye
        (xr, yr) = r_eye
        (xml, yml) = mouth_l
        (xmr, ymr) = mouth_r
        
        (xn, yn) = ( 0.5*(xl+xr), 0.5*(yl+yr) )
        (xm, ym) = ( 0.5*(xml+xmr), 0.5*(yml+ymr) )
        (dx, dy) = (xm-xn, ym-yn)
        (xl2, yl2) = (xl+1.1*dx, yl+1.1*dy)
        (xr2, yr2) = (xr+1.1*dx, yr+1.1*dy)
        
        s = self.size
        pts1 = np.float32([[xl, yl], [xr, yr], [xr2, yr2], [xl2, yl2]])
        pts2 = np.float32([[s*0.3, s*0.3], [s*0.7, s*0.3], [s*0.7, s*0.75], [s*0.3, s*0.75]])
        
        return pts1, pts2
