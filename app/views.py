from app import app
from flask import request, redirect, jsonify
from flask import render_template
from werkzeug.utils import secure_filename
import sys
import os
from pprint import pprint
from imageai.Detection import ObjectDetection
from .core.classify_dog_breed import predict

app.config["IMAGE_UPLOADS"] = "/home/duongnam/Desktop/python-service/app/app/core/detect_object/input_data"
app.config["IMAGE_STATIC"] = 'app/static/img/object_detected'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

@app.route("/")
def index():
    return render_template("public/index.html")


def allowed_image(filename):
    
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":
        if request.files:
            if "filesize" in request.cookies:
                if not allowed_image_filesize(request.cookies["filesize"]):
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)

                image = request.files["image"]

                if image.filename == "":
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename):

                    execution_path = os.getcwd()

                    filename = secure_filename(image.filename)
                    # image_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
                    uploaded_path = os.path.join(execution_path, app.config["IMAGE_STATIC"], 'input', filename)
                    image.save(uploaded_path)

                    detector = ObjectDetection()
                    detector.setModelTypeAsRetinaNet()
                    detector.setModelPath( os.path.join(execution_path , 'app/core/detect_object/detect_model/', "resnet50_coco_best_v2.0.1.h5"))
                    # detector.setModelPath( os.path.join(execution_path , 'app/core/detect_object/detect_model/', "yolo-tiny.h5"))
                    detector.loadModel()

                    detections = detector.detectObjectsFromImage(input_image=uploaded_path, 
                    output_image_path=os.path.join(execution_path, app.config["IMAGE_STATIC"], 'output/' + filename ), extract_detected_objects=True)

                    accepted_dogs = filter_detect_dog(detections)
                    # return jsonify(detections)
                    dog_predictor = predict.DogBreedPrediction(accepted_dogs)
                    results = dog_predictor.predict()

                    img_public_path = os.path.join('static', 'img/object_detected')
                    data_final = {}
                    for i in range(len(accepted_dogs)):
                        data_final[os.path.join(img_public_path, accepted_dogs[i][73:])] = results[i]
                    show_uploaded_path = uploaded_path[46:]
                    return render_template("public/result_detect.html", results=data_final, show_uploaded_path=show_uploaded_path )
                else:
                    print("That file extension is not allowed")
                    return redirect(request.url)

    return render_template("public/upload_image.html")


def filter_detect_dog(detection):
    result = []
    for i, image in enumerate(detection[0]):
        if image["percentage_probability"] > 90 and image["name"] == "dog":
            result.append(detection[1][i])
    return result
                
