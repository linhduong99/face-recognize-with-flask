from utils import *


def _input_name():
    print('Enter person name(no space): ')
    name = input()
    if name.isalpha():
        return name
    else:
        print('Invalid Input')
        del name
        _input_name()


# Video Web Face Recognizer
class VideoWFR:    
    def __init__(self, detector, sender):
        self.detector = detector
        self.sender = sender
    
    def process(self, image_url, align=False, save_path=None):
        detection_num = 0
        rec_num = 0
        print('image_url', image_url)
        img = cv2.imread(image_url)
        print("img__", type(img))
        d_name = 'AI face recognition'
        cv2.namedWindow(d_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(d_name, 960, 720)

        dt = 0
        t1 = time.time()
        faces = self.detector.detect(img)
        f_count = len(faces)
        detection_num += f_count

        names = None
        if f_count > 0 and self.sender is not None:
            names = [None] * f_count
            for (i, face) in enumerate(faces):
                if align:
                    face_align = FaceAlignMouth(160)
                    (f_cropped, f_img) = face_align.align(img, face)
                else:
                    (f_cropped, f_img) = self.detector.extract(img, face)
                if f_img is not None and f_img.size != 0:
                    response = self.sender.send_image_recognize(f_img)
                    print(response["message"])
                    if response["message"] == "RECOGNIZED":
                        name = response["name"]
                        percent = int(response["percent"])
                        if save_path is not None:
                            ps = name + "_" + ("%03d" % percent) + ".png"
                            ps = os.path.join(save_path, ps)
                            cv2.imwrite(ps, f_img)
                        print(response["name"] + ": " + response["percent"])
                    else:
                        # UNRECOGNIZED : add new face or no
                        # print('Do you want to add a new face? ')
                        # print('1.YES')
                        # print('2.NO')
                        # number = input()
                        # print("number", number)
                        # if number == 1:
                        name = _input_name()
                        res = self.sender.send_image_create(name=name, img=img)
                        print(res)

        t2 = time.time()
        dt = dt + (t2 - t1)

        if len(faces) > 0:
            Utils.draw_faces(faces, (0, 0, 255), img, True, True, names)

        # Display the resulting frame
        cv2.imshow(d_name, img)
        cv2.waitKey(0)

        # Capture all frames
        # while(True):
        #     (ret, frame) = capture.read()
        #     if frame is None:
        #         break
        #     frame_count = frame_count+1
        #
        #     t1 = time.time()
        #     faces = self.detector.detect(frame)
        #     f_count = len(faces)
        #     detection_num += f_count
        #
        #     names = None
        #     if (f_count>0) and (not (self.sender is None)):
        #         names = [None]*f_count
        #         for (i, face) in enumerate(faces):
        #             if align:
        #                 (f_cropped, f_img) = fa.align(frame, face)
        #             else:
        #                 (f_cropped, f_img) = self.detector.extract(frame, face)
        #             if (not (f_img is None)) and (not f_img.size==0):
        #                 response = self.sender.send(f_img)
        #                 is_recognized = response["message"] == "RECOGNIZED"
        #                 print(response["message"])
        #                 if is_recognized:
        #                     print(response["name"]+": "+response["percent"])
        #
        #                 if is_recognized:
        #                     rec_num += 1
        #                     name = response["name"]
        #                     percent = int(response["percent"])
        #                     conf = percent*0.01
        #                     names[i] = (name, conf)
        #                     if not (save_path is None):
        #                         ps = ("%03d" % rec_num)+"_"+name+"_"+("%03d" % percent)+".png"
        #                         ps = os.path.join(save_path, ps)
        #                         cv2.imwrite(ps, f_img)
        #
        #     t2 = time.time()
        #     dt = dt + (t2-t1)
        #
        #     if len(faces)>0:
        #         Utils.draw_faces(faces, (0, 0, 255), frame, True, True, names)
        #
        #     # Display the resulting frame
        #     cv2.imshow(dname,frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        # capture.release()
        # cv2.destroyAllWindows()
        
        if dt > 0:
            fps = detection_num/dt
        else:
            fps = 0
        
        return detection_num, rec_num, fps


if __name__ == "__main__":
    image_url = str(sys.argv[1])
    #host = str(sys.argv[2])
    #port = int(sys.argv[3])

    host = "http://127.0.0.1"
    port = 5000
    
    # Video Web recognition 
    save_path = r"rec"
    d = MTCNNDetector(50, 0.95)
    sender = ImgSend(host, port, True)
    vr = VideoWFR(d, sender)

    (f_count, rec_count, fps) = vr.process(image_url=image_url, align=True, save_path=save_path)

    print("Face detections: " + str(f_count))
    print("Face recognitions: " + str(rec_count))
    print("FPS: " + str(fps))
