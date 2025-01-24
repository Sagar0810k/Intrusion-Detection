from flask import Flask, render_template, Response, request
import cv2
import torch
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

intrusion_count = 0  # Global variable to count intrusions

def detect_intrusion(frame):
    """Detect intrusions using YOLOv5 model."""
    global intrusion_count
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get detection results as a pandas DataFrame
    
    intrusion_detected = False  # Flag to indicate if any intrusion is detected
    for _, detection in detections.iterrows():
        if detection['confidence'] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            label = detection['name']
            # Draw bounding box and label for each detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {detection['confidence']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            intrusion_detected = True  # Set the flag as an intrusion is detected
    
    if intrusion_detected:
        intrusion_count += 1
    
    return frame, intrusion_detected  # Return frame with all bounding boxes and the detection flag

def generate_frames(video_path):
    """Generate video frames and process for intrusion detection."""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, intrusion_detected = detect_intrusion(frame)
        if intrusion_detected:
            print(f"Intrusion detected! Total count: {intrusion_count}")
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

def generate_live_frames():
    """Generate video frames from the live camera and process for intrusion detection."""
    cap = cv2.VideoCapture(0)  # Use the default camera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, intrusion_detected = detect_intrusion(frame)
        if intrusion_detected:
            print(f"Intrusion detected! Total count: {intrusion_count}")
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    video_file = request.files['video']
    video_path = "uploaded_video.mp4"
    video_file.save(video_path)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_feed')
def live_feed():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
