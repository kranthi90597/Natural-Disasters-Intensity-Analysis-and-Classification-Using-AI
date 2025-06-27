from flask import Flask, render_template, Response
import cv2  # OpenCV library
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load the model
try:
    model = load_model('disaster.h5')
    print("Loaded model from disk")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to capture webcam frames and process them
def webcam_feed():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return "Error accessing webcam"
    
    index = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Preprocess the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (64, 64))
        x = np.expand_dims(frame_resized, axis=0)

        # Predict disaster type
        result = np.argmax(model.predict(x), axis=-1)
        detected_activity = str(index[result[0]])

        # Add prediction text to the frame
        cv2.putText(frame, f"Activity: {detected_activity}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        
        # Return the frame as a byte stream for video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    # Release the webcam
    cap.release()

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/intro', methods=['GET'])
def about():
    return render_template('intro.html')

@app.route('/upload', methods=['GET'])
def upload():
    return render_template("upload.html")

@app.route('/team', methods=['GET'])
def team():
    return render_template('team.html')


@app.route('/video_feed')
def video_feed():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
