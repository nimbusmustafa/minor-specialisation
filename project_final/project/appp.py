from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import threading
import os
import time
from handwriting_recognition import predict_text
from ultralytics import YOLO
import numpy as np
from PIL import Image
from io import BytesIO
from stl import mesh

app = Flask(__name__)

phone_stream_url = 'http://10.87.2.81:8080/video'
webcam = cv2.VideoCapture(2)

captured_text = ""
loaded_stl_model = None
mode = "handwriting"  # default mode is handwriting
is_webcam_feed = False  # flag to control webcam feed
loaded_video_cap = None  # OpenCV VideoCapture for the letter video
gif_frames = []  # Store frames of the GIF
gif_frame_index = 0  # Current index for the GIF frame

gesture_model = YOLO('/home/mustafa/Downloads/Handwritten-Text-Recognition-master/project/bestv11n.pt')

camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

# Marker size in meters
marker_size = 0.1  # 1 cm marker size

# Open the webcam

def load_gif(gif_path):
    global gif_frames
    gif = Image.open(gif_path)
    gif_frames = []
    while True:
        gif_frames.append(np.array(gif.convert('RGB')))
        try:
            gif.seek(gif.tell() + 1)
        except EOFError:
            break

def render_3d_model_on_marker(frame, stl_model, rvec, tvec, camera_matrix, dist_coeffs, scale_factor):
    object_points = np.array([vertex for facet in stl_model.vectors for vertex in facet], dtype=np.float32) * scale_factor
    imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    for i in range(0, len(imgpts), 3):
        cv2.polylines(frame, [imgpts[i:i+3]], isClosed=True, color=(0, 255, 0), thickness=2)

def generate_phone_feed():
    while True:
        cap = cv2.VideoCapture(phone_stream_url)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from phone feed")
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_webcam_feed():
    global gif_frame_index
    while True:
        ret, frame = webcam.read()
        if not ret:
            continue

        if mode == "gesture":
            if loaded_stl_model is not None:
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, _ = detector.detectMarkers(frame)

                if ids is not None:
                    print("Detected marker IDs:", ids.flatten())
                    for i, marker_id in enumerate(ids.flatten()):
                        if marker_id == 0:
                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                [corners[i]], marker_size, camera_matrix, dist_coeffs
                            )
                            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                            render_3d_model_on_marker(
                                frame, loaded_stl_model, rvecs[0], tvecs[0],
                                camera_matrix, dist_coeffs, scale_factor=0.001
                            )
                            print(f"Rendered {captured_text}.stl on the webcam feed.")

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        elif mode == "handwriting":
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(frame)

            if ids is not None and gif_frames:
                marker_corners = corners[0][0].astype(np.float32)

                if len(marker_corners) == 4:
                    gif_frame = gif_frames[gif_frame_index % len(gif_frames)]
                    gif_frame_index += 1

                    # -------- Offset and Scaling settings --------
                    offset_x =  600   # pixels to move right (+) or left (-)
                    offset_y =  600  # pixels to move down (+) or up (-)
                    scale_factor = 4  # 1.0 = original size, >1 = bigger, <1 = smaller
                    # ------------------------------------------------

                    h, w, _ = gif_frame.shape

                    # Apply scaling
                    new_w = int(w * scale_factor)
                    new_h = int(h * scale_factor)
                    gif_frame_resized = cv2.resize(gif_frame, (new_w, new_h))

                    # New source points after scaling
                    src_pts = np.array([
                        [offset_x, offset_y],
                        [offset_x + new_w - 1, offset_y],
                        [offset_x + new_w - 1, offset_y + new_h - 1],
                        [offset_x, offset_y + new_h - 1]
                    ], dtype=np.float32)

                    dst_pts = marker_corners

                    # Find homography matrix
                    M, _ = cv2.findHomography(src_pts, dst_pts)

                    # Warp the resized gif onto the marker
                    warped_gif = cv2.warpPerspective(gif_frame_resized, M, (frame.shape[1], frame.shape[0]))

                    # Create mask from the warped gif
                    gif_mask = cv2.cvtColor(warped_gif, cv2.COLOR_BGR2GRAY)
                    _, gif_mask = cv2.threshold(gif_mask, 1, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(gif_mask)

                    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                    gif_fg = cv2.bitwise_and(warped_gif, warped_gif, mask=gif_mask)

                    frame = cv2.add(frame_bg, gif_fg)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    global captured_text, mode, is_webcam_feed
    return render_template('index.html', prediction=captured_text, mode=mode, is_webcam_feed=is_webcam_feed)

@app.route('/switch_to_phone', methods=['POST'])
def switch_to_phone():
    global is_webcam_feed
    is_webcam_feed = False  # Switch back to phone feed
    return redirect(url_for('index'))

@app.route('/webcam')
def webcam_feed():
    return Response(generate_webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/phone')
def phone_feed():
    return Response(generate_phone_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_from_phone():
    global captured_text, mode, is_webcam_feed, loaded_stl_model, loaded_video_cap

    if not os.path.exists('gesture'):
        os.makedirs('gesture')
    if not os.path.exists('handwriting'):
        os.makedirs('handwriting')

    cap = cv2.VideoCapture(phone_stream_url)
    ret, frame = cap.read()

    if ret:
        if mode == "handwriting":
            folder = 'handwriting'
        elif mode == "gesture":
            folder = 'gesture'
        else:
            folder = 'other'

        filename = f"{folder}/captured_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)

        if mode == "handwriting":
            predicted = predict_text(filename)
            print(f"Predicted text from handwriting: {predicted}")
            captured_text = predicted.upper()  # Make sure it's uppercase for GIF filenames

            gif_path = f'videos/{captured_text}.gif' 
            print(gif_path) # GIF corresponding to the predicted letter
            if os.path.exists(gif_path):
                load_gif(gif_path)
                loaded_video_cap = None
                loaded_stl_model = None
                print("loaded")
            else:
                print("GIF file not found.")
                loaded_video_cap = None

        elif mode == "gesture":
            results = gesture_model.predict(filename, conf=0.5)
            if len(results[0].boxes) > 0:
                predicted = results[0].names[int(results[0].boxes.cls[0])]
                print(f"Predicted gesture: {predicted}")
                captured_text = predicted.upper()
            else:
                predicted = "No gesture detected"
                captured_text = predicted
                print("No gesture detected.")

            stl_path = f'stl_models/letters_stl/{captured_text}.stl'
            if os.path.exists(stl_path):
                loaded_stl_model = mesh.Mesh.from_file(stl_path)
                loaded_video_cap = None
            else:
                loaded_stl_model = None

        is_webcam_feed = True

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
    app.run(debug=True, use_reloader=False)
