from flask import Flask, request
import json
# import pyrebase  #pip install pyrebase4

import numpy as np
import cv2

from code_pattern1 import process


app = Flask(__name__)

# @app.route("/processing", methods=["POST"])
# def index():
#     # Image
#     imagefile = request.files["image"].read()


#     npimg = np.fromstring(imagefile, np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     # image = Image.fromarray(img)
    

#     error, x_point = process(img)
#     print("point"+ str(x_point))


#     return json.dumps({"error_img": error.tolist(), "score": x_point})

@app.route('/')
def home():
    return "Hello World"


if __name__ == '__main__':
    app.run(debug=True)