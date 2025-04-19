from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import threading
import os
from handwriting_recognition import predict_text
from ultralytics import YOLO

app = Flask(__name__)

phone_stream_url = 'http://10.87.2.81:8080/video'
webcam = cv2.VideoCapture(0)

captured_text = ""
mode = "handwriting"  # default mode is handwriting

# Load YOLO model for gesture/alphabet recognition
gesture_model = YOLO('/home/mustafa/Downloads/Handwritten-Text-Recognition-master/project/bestv11n.pt')  # <-- change this path

def generate_webcam_feed():
    while True:
        ret, frame = webcam.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_phone_feed():
    cap = cv2.VideoCapture(phone_stream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    global captured_text, mode
    return render_template('index.html', prediction=captured_text, mode=mode)

@app.route('/webcam')
def webcam_feed():
    return Response(generate_webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/phone')
def phone_feed():
    return Response(generate_phone_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_from_phone():
    global captured_text, mode
    cap = cv2.VideoCapture(phone_stream_url)
    ret, frame = cap.read()
    if ret:
        save_path = "captured.jpg"
        cv2.imwrite(save_path, frame)

        if mode == "handwriting":
            predicted = predict_text(save_path)
        elif mode == "gesture":
            results = gesture_model.predict(save_path, conf=0.5)
            if len(results[0].boxes) > 0:
                # Assuming you want the first detected class
                predicted = results[0].names[int(results[0].boxes.cls[0])]
            else:
                predicted = "No gesture detected"
        
        captured_text = predicted

    cap.release()
    return redirect(url_for('index'))

@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    global mode
    if mode == "handwriting":
        mode = "gesture"
    else:
        mode = "handwriting"
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
