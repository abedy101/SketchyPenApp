from flask import Flask, render_template, request, send_file
from PIL import Image
import numpy as np
import cv2
import os
from io import BytesIO

app = Flask(__name__)

def pencil_sketch(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    # Invert the blurred image
    inverted_blurred_image = 255 - blurred_image
    # Combine the grayscale image with the inverted blurred image using the "color dodge" blend mode
    pencil_sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
    return pencil_sketch_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file:
        # Read image file
        image_stream = file.read()
        # Convert image stream to numpy array
        np_image = np.frombuffer(image_stream, np.uint8)
        # Decode numpy array to OpenCV BGR format
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        # Perform pencil sketch conversion
        sketch_image = pencil_sketch(image)
        # Encode sketch image to JPEG format
        _, sketch_image_stream = cv2.imencode('.jpg', sketch_image)
        # Convert sketch image stream to bytes
        sketch_image_bytes = sketch_image_stream.tobytes()
        # Create BytesIO object to store image bytes
        sketch_image_io = BytesIO(sketch_image_bytes)
        # Return sketch image file
        return send_file(sketch_image_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
