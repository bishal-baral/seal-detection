from flask import Flask, render_template, request
import requests
import cv2
from PIL import Image
import io, base64
import numpy as np
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test_process_image')
def test_process_image():
    return render_template('result.html', original_image=""
                           , all_data=[[1078.0, 68.0, 1312.0, 251.0, 0.0, 0.8906891942024231, (43, 50, 41)], [200, 200, 200, 200, 1.0, 0.2932696044445038, (77, 57, 97)]])


@app.route('/process_image', methods=['POST'])
def process_image():
    # default_value = '0'
    # data = request.form.get('photo', default_value)
    image_text = request.form['image_text']
    # print(image_text)
    response = requests.post("https://bbaral-chinese-seal-detector.hf.space/run/predict", json={
        "data": [
            image_text,
        ]
    }).json()

    # result = response["image"]
    print(response.keys())
    print(len(response["data"]))
    result = response["data"]

    image = result[0]
    all_data = result[1]['data']
    print(result[1]['data'])

    # Convert the base64 string to a bytes string
    imgdata = base64.b64decode(image_text.split(',')[1])

    # Convert the bytes string to an OpenCV image
    cv_img = cv2.imdecode(np.fromstring(imgdata, np.uint8), cv2.IMREAD_COLOR)


    for i, d in enumerate(all_data):
        # box = np.array(d[:4]).astype(int)
        x1, y1, x2, y2 = np.array(d[:4]).astype(int)
        conf = round(d[5], 2)

        # Generate a random color for the box and label background
        color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        all_data[i].append(color)
        # Draw the box
        label = f'Object {i}, {conf}'
        # Draw the label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(cv_img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0] +5, y1), color, -1)
        cv2.putText(cv_img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 4)

        # # Add labels to the boxes
        # cv2.putText(cv_img, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Convert the modified image to a bytes string
    retval, buffer = cv2.imencode('.jpg', cv_img)
    jpg_as_text = base64.b64encode(buffer)

    # Convert the bytes string to a base64 string
    base64_image_out = jpg_as_text.decode('utf-8')
    
    return render_template('result.html', original_image=base64_image_out, all_data=all_data)

if __name__ == '__main__':
    app.run(debug=True)