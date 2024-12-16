from flask import Flask, session, render_template, Response, jsonify
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from flask_cors import CORS
import winsound
app = Flask(__name__)
app.secret_key = '1234' # Secures the client-side data by signing it in the session (secures sending commands, stream, etc.)
CORS(app) # Allows cross-origin requests (different ports can be used in implementations to reduce traffic)

# URLs for ESP32 and video stream
ESP32_URL = "http://192.168.4.1"
STREAM_URL = "http://192.168.4.1:81/stream"

# Load YOLO models for object and lane detection
model = YOLO("yolov8n.pt")
lane_model = YOLO("runs/detect/train/weights/best.pt")

is_autonomous = False
is_object = False

@app.before_request
def ensure_session_state():
    global is_autonomous
    if 'is_autonomous' not in session:
        session['is_autonomous'] = False
    is_autonomous = session['is_autonomous']

@app.route('/toggle_autonomous', methods=['POST'])
def toggle_autonomous():
    global is_autonomous
    is_autonomous = not is_autonomous # Toggles is_autonomous (True -> False, False -> True)
    session['is_autonomous'] = is_autonomous # Store the state in the session
    status = "enabled" if is_autonomous else "disabled"
    print(f"Autonomous mode is now {status}.")
    return jsonify({"autonomous_mode": status}), 200 # Return status code (200 = OK, 400 = BAD, 404 = Not Found)

# -------------------- DETECTION METHODS -------------------- #

def detect_objects(frame):
    global is_object
    is_object = False # Make sure each object detection check begins with 'False'
    height, width, _ = frame.shape # Frame dimensions
    results = model.predict(frame, conf=0.5, show=False) # 0.5 confidence threshold for objects in frame (increase or decrease for sensitivity/accuracy)

    for box in results[0].boxes: # Iterate through the detected bounding boxes
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # Convert coordinates of box to integers
        conf = float(box.conf[0])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Draw the box on the frame
        label = f"{conf * 100:.1f}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        obj = detect_and_stop(x1, x2, y1, y2, frame) # Check if the detected object is within the region of interest (ROI)
        if obj is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw box as red if it is in the ROI
            cv2.putText(frame, "IN ROI", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print("Object detected in ROI! Stopping the car.")
            is_object = True
        else:
            is_object = False


def detect_and_stop(x1, x2, y1, y2, frame):
    height, width, _ = frame.shape # Frame dimensions

    # Define the ROI for if an object is in it, the car will stop (i.e. top, bottom, left, right are all distances from the frame edges)
    roi_top = int(height * 0.4)
    roi_bottom = height
    roi_left = int(width * 0.4)
    roi_right = int(width * 0.6)

    if not (x2 < roi_left or x1 > roi_right or y2 < roi_top or y1 > roi_bottom): # Check if bounding box of detected object intersects with ROI
        send_command("STOP")
        winsound.PlaySound("*", winsound.SND_ALIAS) # Plays Windows sound
        return True


def detect_lanes(frame, frame_width):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert to HSV (better than RGB)

    # Set HSV values for an upper and lower bound of black (road color), along with upper and lower bound for white (lane color)
    # Masks preprocess the image to isolate lanes and roads
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_black = cv2.inRange(hsv, lower_black, upper_black) # Mask all black colors in bounds
    mask_white = cv2.inRange(hsv, lower_white, upper_white) # Mask all white colors in bounds
    mask = cv2.bitwise_or(mask_black, mask_white) # Combine black and white masks

    # --- COMPUTER VISION (MODEL-BASED DETECTION) --- #
    results = lane_model.predict(frame, conf=0.9, show=False) # Predict lane lines over 0.9 confidence

    left_lane_lines = []
    right_lane_lines = []
    if results[0].boxes is not None: # Check for bounding boxes detected by lane model
        for box in results[0].boxes: # Iterate through the detected bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf') # Calculate slope of the line ( | , / , \ , etc. directions)
            if slope > 1 and x1 > frame_width // 2: # Determine left lanes based on slope direction
                left_lane_lines.append((x1, y1, x2, y2, conf))
            elif slope < -1 and x1 < frame_width // 2: # Determine right lanes based on slope direction
                right_lane_lines.append((x1, y1, x2, y2, conf))

    # --- COMPUTER VISION (EDGE-BASED DETECTION) --- #
    # Use to minimize false positives, improve accuracy, etc. to make up for smaller YOLO trained sample size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Greyscale frame for edge detection
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 100, 200) # Use Gaussian blue and Canny edge detection
    height, width = frame.shape[:2] # Frame parameters

    # Mask the ROI ('roi_mask' shows the region)
    mask = np.zeros_like(edges)
    polygon = np.array([[ # ROI logic similar to object detection, but uses /  \ shape here similar to how a road is from the car's perspective)
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.7), int(height * 0.6)),
        (int(width * 0.3), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    cropped_edges = cv2.bitwise_and(edges, mask) # Mask the edges to detect within ROI
    hough_lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50) # Hough transform to extract lanes
    if hough_lines is not None:
        for line in hough_lines: # Iterate through Hough lines (same as with the YOLO detected lanes)
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            if slope > 1 and x1 > frame_width // 2:
                left_lane_lines.append((x1, y1, x2, y2, 0.0))
            elif slope < -1 and x1 < frame_width // 2:
                right_lane_lines.append((x1, y1, x2, y2, 0.0))
    left_innermost = None
    right_innermost = None

    # Since lanes have 2 edges per each (i.e. ||   ||) only use the inner most for best lane center accuracy
    if left_lane_lines:
        left_innermost = min(left_lane_lines, key=lambda line: abs((line[0] + line[2]) // 2 - frame_width // 2))
    if right_lane_lines:
        right_innermost = min(right_lane_lines, key=lambda line: abs((line[0] + line[2]) // 2 - frame_width // 2))
    return [left_innermost] if left_innermost else [], [right_innermost] if right_innermost else []

# ---------------------------------------------------------- #

def calculate_lane_center(left_lane_points, right_lane_points, frame_width):
    try:
        left_x = [x1 for x1, _, _, _, _ in left_lane_points] # Take left lane x-coordinates
        right_x = [x1 for x1, _, _, _, _ in right_lane_points] # Take right lane x-coordinates
        if left_x and right_x:
            lane_center = (np.mean(left_x) + np.mean(right_x)) // 2 # Take average of means (accounts for inaccuracies)
        elif left_x:
            lane_center = np.mean(left_x) + frame_width // 4 # Only left, therefore add 1/4 of the frame width to account for centering
        elif right_x:
            lane_center = np.mean(right_x) - frame_width // 4 # Same logic as above, but for right
        else:
            lane_center = None
        print("Calculated Lane Center:", lane_center)
        return int(lane_center) if lane_center is not None else None
    except Exception as e:
        print(f"Error calculating lane center: {e}")
        return None


def process_frame(frame):
    frame_width = frame.shape[1] # Get the width
    left_lane_points, right_lane_points = detect_lanes(frame, frame_width) # Detect the lanes
    if left_lane_points is not None and right_lane_points is not None:
        lane_center = calculate_lane_center(left_lane_points, right_lane_points, frame_width)
        frame_center = frame_width // 2

    tolerance = 64 # Allows for a 20% offset of the car before sending commands (tolerance = 32 or less is ideal for general use-cases of a larger car)
    print("Frame Center:", frame_center, "Lane Center:", lane_center)

    # --- DRAW THE LANE LINES --- #
    drive(lane_center, frame_center, tolerance, left_lane_points, right_lane_points) # Based on the lane center and frame center, send drive commands for FORWARD, BACK, LEFT, RIGHT, STOP
    lane_overlay = np.zeros_like(frame) # Create the overlay to draw lane lines
    for x1, y1, x2, y2, _ in left_lane_points: # Draw left lane lines in green
        cv2.line(lane_overlay, (x1, y1), (x2, y2), (0, 255, 0), 20)

    for x1, y1, x2, y2, _ in right_lane_points: # Draw right lane lines in blue
        cv2.line(lane_overlay, (x1, y1), (x2, y2), (255, 0, 0), 20)

    if lane_center: # Draw a circle at the lane center in yellow
        cv2.circle(lane_overlay, (lane_center, frame.shape[0] // 2), 5, (0, 255, 255), -1)
    blended_frame = cv2.addWeighted(frame, 0.8, lane_overlay, 0.5, 1)
    return blended_frame

def drive(lane_center, frame_center, tolerance, left_lane_points, right_lane_points):
    if lane_center is not None:
        # Based on the tolerance and offset of lane vs. frame center, send directions to the ESP32
        offset = lane_center - frame_center
        if abs(offset) > tolerance:
            direction = "LEFT" if offset < 0 else "RIGHT"
            print(f"Command: {direction}")
            send_command(direction)
        else:
            print("Command: FORWARD")
            send_command("FORWARD")
    else:
        print("Command: STOP")
        send_command("STOP")


def send_command(command):
    global is_autonomous
    if is_autonomous: # If in autonomous mode, try to send the drive commands to the ESP32
        try:
            response = requests.get(f"{ESP32_URL}/control?direction={command}")
            print(f"Sent command: {command}, Response: {response.status_code}")
        except Exception as e:
            print(f"Error sending command: {e}")

@app.route('/')
def index():
    return render_template('index.html')

# REFERENCE: https://community.openhab.org/t/usage-of-image-item/51462/3
@app.route('/processed_stream')
def processed_stream():
    return Response(generate_processed_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_processed_stream():
    response = requests.get(STREAM_URL, stream=True)
    if response.status_code == 200: # OK status
        bytes_data = b'' # Initialize byte string for data

        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk # Append chunk to bytes_data
            a = bytes_data.find(b'\xff\xd8') # Start of jpg image in byte stream
            b = bytes_data.find(b'\xff\xd9') # End of jpg image in the byte stream
            if a != -1 and b != -1: # Both start and end found
                jpg = bytes_data[a:b + 2]  # Extract the JPEG from byte stream
                bytes_data = bytes_data[b + 2:] # Remove processed from byte stream
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR) # Decode to OpenCV

                if frame is not None:
                    detect_objects(frame) # First detect objects
                    if is_object:
                        processed_frame = frame
                    else:
                        processed_frame = process_frame(frame) # Then detect lanes
                    ret, buffer = cv2.imencode('.jpg', processed_frame) # Encode back to jpg

                   # Display frame after encoding
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    else:
        print(f"Failed to access stream. Status code: {response.status_code}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
