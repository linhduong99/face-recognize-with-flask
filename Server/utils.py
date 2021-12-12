import os
import numpy as np
import cv2
from keras.models import load_model
from scipy.spatial import distance


class Utils:
    @staticmethod
    def draw_face(face, color, frame, draw_points=True, draw_rect=True, n_data=None):
        (x1, y1, w, h) = face['box']
        confidence = face['confidence']
        x2 = x1 + w
        y2 = y1 + h
        if draw_rect:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        y3 = y1 - 12
        if not (n_data is None):
            (name, conf) = n_data
            text = name + (" %.3f" % conf)
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
        res = (x + dx, y + dy)
        return res

    @staticmethod
    def draw_point(point, color, frame):
        (x, y) = point
        x1 = x - 1
        y1 = y - 1
        x2 = x + 1
        y2 = y + 1
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
        for (i, fname) in enumerate(filenames):
            fullpath = os.path.join(folder, fname)
            files.append(fullpath)
        return files


# Face Database
class FaceDB:
    def __init__(self):
        self.clear()

    def clear(self):
        self.f_data = []

    def load(self, db_path, rec):
        self.clear()
        files = FileUtils.get_files(db_path)
        for (i, fname) in enumerate(files):
            f_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            embds = rec.embeddings(f_img)
            f = os.path.basename(fname)
            p_name = os.path.splitext(f)[0]
            data = (p_name, embds, f_img)
            self.f_data.append(data)

    def get_data(self):
        return self.f_data


# FaceNet recognizer
class FaceNetRec:
    def __init__(self, model, min_distance):
        self.model = load_model(model)
        self.min_distance = min_distance

    def get_model(self):
        return self.model

    def embeddings(self, f_img):
        r_img = cv2.resize(f_img, (160, 160), cv2.INTER_AREA)
        arr = r_img.astype('float32')
        arr = (arr - 127.5) / 127.5
        samples = np.expand_dims(arr, axis=0)
        embds = self.model.predict(samples)
        return embds[0]

    def eval_distance(self, embds1, embds2):
        dist = distance.cosine(embds1, embds2)
        return dist

    def img_distance(self, f_img1, f_img2):
        embds1 = self.embeddings(f_img1)
        embds2 = self.embeddings(f_img2)
        dist = self.eval_distance(embds1, embds2)
        return dist

    def match(self, embds1, embds2):
        dist = self.eval_distance(embds1, embds2)
        return dist <= self.min_distance

    def img_match(self, f_img1, f_img2):
        embds1 = self.embeddings(f_img1)
        embds2 = self.embeddings(f_img2)
        return self.match(embds1, embds2)

    def recognize(self, embds, f_db):
        minfd = 2.0
        indx = -1
        f_data = f_db.get_data();
        for (i, data) in enumerate(f_data):
            (name, embds_i, p_img) = data
            dist = self.eval_distance(embds, embds_i)
            if (dist < minfd) and (dist < self.min_distance):
                indx = i
                minfd = dist
        if indx >= 0:
            (name, embds_i, p_img) = f_data[indx]
            return (name, minfd, p_img)

        return None

    def img_recognize(self, f_img, f_db):
        embds = self.embeddings(f_img)

        return self.recognize(embds, f_db)


class RecData:
    def __init__(self):
        self.rec_count = 0

    def count(self):
        self.rec_count += 1

    def get_count(self):
        return self.rec_count
