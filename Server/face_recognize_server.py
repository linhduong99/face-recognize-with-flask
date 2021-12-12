import json
import time
import jsonpickle
import tensorflow as tf
import keras
import flask
from flask import Flask, request, Response
from utils import *


print("cv2_version : {}".format(cv2.__version__))
print("tf_version : {}".format(tf.__version__))
print("keras_version : {}".format(keras.__version__))
print("flask_version : {}".format(flask.__version__))

# Initialize the Flask application
app = Flask(__name__)

rec = None
f_db = None
rec_data = None
save_path = None


@app.route("/create", methods=['GET', 'POST'])
def create_face_image():
    name = request.json["name"]
    image_bytes = bytes(request.json["image"], 'ISO-8859-1')
    np_array_img = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(np_array_img, cv2.IMREAD_COLOR)
    if img is not None:
        path = "F:\Code\Server\db\{}.png".format(name)
        cv2.imwrite(path, img)
        response = {'message': 'Successfully added!'}
        status_code = 200
    else:
        response = {'message': 'Add failed!'}
        status_code = 404
    print(response)
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=status_code, mimetype="application/json")


@app.route("/face-recognize", methods=['POST'])
def face_recognize():
    print("Processing recognition request... ")
    t1 = time.time()
    np_array_img = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(np_array_img, cv2.IMREAD_COLOR)
    embds = rec.embeddings(img)
    data = rec.recognize(embds, f_db)
    print('data', data)
    t2 = time.time()
    dt = t2-t1
    print("Recognition request processed: " + str(dt) + " sec")
    
    rec_data.count()
    if data is not None:
        (name, dist, p_photo) = data
        conf = 1.0 - dist
        percent = int(conf*100)
        info = "Recognized: " + name + " " + str(conf)
        ps = ("%03d" % rec_data.get_count()) + "_" + name + "_" + ("%03d" % percent) + ".png"
        response = {
            "message": "RECOGNIZED",
            "name": name,
            "percent": str(percent)}
    else:
        info = "UNRECOGNIZED"
        ps = ("%03d" % rec_data.get_count()) + "_unrecognized" + ".png"
        response = {"message": "UNRECOGNIZED"}
    
    print(info)
    if save_path is not None:
       ps = os.path.join(save_path, ps)
       cv2.imwrite(ps, img)
        
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    # host = str(sys.argv[1])
    # port = int(sys.argv[2])

    # FaceNet recognizer
    m_file = r"models/facenet_keras.h5"
    rec = FaceNetRec(m_file, 0.5)
    rec_data = RecData()
    print("Recognizer loaded.")
    print(rec.get_model().inputs)
    print(rec.get_model().outputs)
    
    # Face DB 
    save_path = r"rec"
    db_path = r"F:\Code\Server\db"
    f_db = FaceDB()
    f_db.load(db_path, rec)
    db_f_count = len(f_db.get_data())
    print("Face DB loaded: " + str(db_f_count))
    
    print("Face recognition running")
          
    host = "127.0.0.1"
    port = 5000
    app.run(host=host, port=port, threaded=False, debug=True)

# END
