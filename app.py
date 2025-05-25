from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, session
import cv2
import dlib
import numpy as np
import sqlite3
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wavfile
import time
from difflib import SequenceMatcher
import easyocr # Using EasyOCR for OCR functionality
import os
import threading # Import threading for potential future async recording

###############################################################################
# Flask App Initialization
###############################################################################
app = Flask(__name__)
app.secret_key = 'your_secret_key' # Replace 'your_secret_key' with a secure key

# Configure upload folder for OCR images
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

###############################################################################
# Dlib Models (Face Detector & Shape Predictor)
###############################################################################
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Make sure "shape_predictor_68_face_landmarks.dat" is in the same directory
# or provide the full path.

# Initialize EasyOCR Reader
###############################################################################
reader = easyocr.Reader(['en']) # Specify language(s)

###############################################################################
# Global Eye Tracking Data
# Stores { "time": <timestamp>, "closed": <bool> } for each frame
###############################################################################
eye_tracking_data = []

###############################################################################
# Database Functions
###############################################################################
def get_db_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates the 'posts' table if it does not exist."""
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                textbody TEXT NOT NULL,
                speech_text TEXT,
                accuracy REAL
            );
        ''')
        conn.commit()

###############################################################################
# Audio Recording & Speech Recognition
###############################################################################
def record_audio(file_name="mic_recording.wav", duration=10, fs=44100):
    """
    Records audio from the default microphone using 'sounddevice' for
    a specified duration and writes it to a WAV file.
    """
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait() # Wait until recording is finished
    wavfile.write(file_name, fs, recording)
    print("Recording complete, saved as", file_name)

def speech_to_text():
    """
    Records audio via record_audio(), then uses Google Speech Recognition
    to convert the recorded WAV file into text.
    """
    recognizer = sr.Recognizer()
    record_audio() # Record audio first

    with sr.AudioFile("mic_recording.wav") as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Recognition error"
        except sr.RequestError as e:
            return f"Service error: {e}"

###############################################################################
# Text Comparison
###############################################################################
def compare_texts(input_text, speech_text):
    """
    Returns a percentage similarity between 'input_text' and 'speech_text'
    using the SequenceMatcher ratio.
    """
    return SequenceMatcher(None, input_text, speech_text).ratio() * 100

###############################################################################
# Eye Tracking (OpenCV + Dlib)
###############################################################################
def detect_eye_state(frame):
    """
    Detects if eyes are closed or open in the given frame.
    Updates the global 'eye_tracking_data' with a timestamp and 'closed' bool.
    """
    global eye_tracking_data

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    eye_closed = False # Default to eyes open

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract left & right eye coordinates
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        def eye_aspect_ratio(eye_points):
            """
            Calculates the Eye Aspect Ratio (EAR) for the given eye landmarks.
            EAR < 0.2 indicates a closed eye (tune threshold as needed).
            """
            A = np.linalg.norm([eye_points[1].x - eye_points[5].x,
                                eye_points[1].y - eye_points[5].y])
            B = np.linalg.norm([eye_points[2].x - eye_points[4].x,
                                eye_points[2].y - eye_points[4].y])
            C = np.linalg.norm([eye_points[0].x - eye_points[3].x,
                                eye_points[0].y - eye_points[3].y])
            return (A + B) / (2.0 * C)

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)

        EAR_THRESHOLD = 0.2
        if left_EAR < EAR_THRESHOLD and right_EAR < EAR_THRESHOLD:
            eye_closed = True

    # Record the eye state with timestamp
    timestamp = time.time()
    eye_tracking_data.append({"time": timestamp, "closed": eye_closed})

    return eye_closed

###############################################################################
# Video Streaming for Eye State Display
###############################################################################
def generate_frames():
    """
    Captures frames from the default camera, applies eye detection,
    draws status (OPEN/CLOSED), and yields them for live streaming.
    """
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Detect eye state
        eye_closed = detect_eye_state(frame)
        color = (0, 0, 255) if eye_closed else (0, 255, 0) # Green = CLOSED, Red = OPEN
        label = "CLOSED" if eye_closed else "OPEN"

        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Yield in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

###############################################################################
# Flask Routes
###############################################################################
@app.route('/')
def index():
    """Renders the main page (index.html)."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Returns a live video feed (multipart/x-mixed-replace).
    Displays eye state in real time.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/eye_tracking_data', methods=['GET'])
def get_eye_tracking_data():
    """
    Returns the eye tracking data (timestamps + closed/open boolean) as JSON.
    """
    return jsonify(eye_tracking_data)

@app.route('/submit_text', methods=['POST'])
def submit_text():
    """
    Receives text input from the user and stores it in the 'posts' table.
    """
    textbody = request.form['textbody']
    with get_db_connection() as conn:
        conn.execute('INSERT INTO posts (textbody) VALUES (?)', (textbody,))
        conn.commit()
    return redirect(url_for('index'))

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """
    Triggers speech recording, updates the last post in DB with recognized speech.
    """
    with get_db_connection() as conn:
        post = conn.execute('SELECT * FROM posts ORDER BY id DESC LIMIT 1').fetchone()
        if post:
            recognized_speech = speech_to_text()
            conn.execute('UPDATE posts SET speech_text = ? WHERE id = ?',
                         (recognized_speech, post['id']))
            conn.commit()
    return redirect(url_for('index'))

@app.route('/results')
def results():
    """
    Calculates similarity accuracy between the stored text and recognized speech,
    updates DB, and renders 'results.html' with final accuracy.
    """
    with get_db_connection() as conn:
        post = conn.execute('SELECT * FROM posts ORDER BY id DESC LIMIT 1').fetchone()
        if post and post['speech_text']:
            accuracy = compare_texts(post['textbody'], post['speech_text'])
            conn.execute('UPDATE posts SET accuracy = ? WHERE id = ?',
                         (accuracy, post['id']))
            conn.commit()
        else:
            accuracy = None
    return render_template('results.html', post=post, accuracy=accuracy)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    filename = file.filename
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)
    try:
        result = reader.readtext(image_path)
        extracted_text = " ".join([r[1] for r in result])
        print(f"Extracted Text: {extracted_text}")
    except Exception as e:
        extracted_text = f"Error extracting text: {str(e)}"
        print(f"OCR Error: {extracted_text}")
    # Save extracted text into database (or update the latest record)
    if extracted_text and not extracted_text.startswith("Error"):
        with get_db_connection() as conn:
            # For simplicity, insert a new record with the extracted text
            conn.execute('INSERT INTO posts (textbody) VALUES (?)', (extracted_text,))
            conn.commit()
    # Save the extracted text in session so it can be displayed on the home page
    session["extracted_text"] = extracted_text
    return redirect(url_for('index'))

@app.route('/clear_text', methods=['POST'])
def clear_text():
    with get_db_connection() as conn:
        conn.execute('UPDATE posts SET textbody = "" WHERE id = (SELECT id FROM posts ORDER BY id DESC LIMIT 1)')
        conn.commit()
    return redirect(url_for('index'))

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text_route():
    """
    Alternate speech recognition endpoint that returns JSON with recognized text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            recognized_text = recognizer.recognize_google(audio)
            return jsonify({"recognized_text": recognized_text})
        except sr.UnknownValueError:
            return jsonify({"recognized_text": "Could not understand audio"})
        except sr.RequestError as e:
            return jsonify({"recognized_text": f"Service error: {e}"})

@app.route('/analyze_results', methods=['POST'])
def analyze_results():
    """
    Analyzes each word in the speech text, checking if eyes were closed (timestamp-based).
    Returns JSON with {"word": <word>, "closed": <bool>}.
    """
    data = request.get_json()
    speech_text = data.get("speech", "")

    if not speech_text:
        return jsonify({"error": "No speech text provided"}), 400

    # Gather all timestamps where eyes were closed
    closed_timestamps = [entry['time'] for entry in eye_tracking_data if entry['closed']]

    result_analysis = []
    # We assume ~0.5s per word. This is an approximation.
    words = speech_text.split()
    start_time = closed_timestamps[0] if closed_timestamps else time.time()
    word_time = start_time

    for word in words:
        # Check if any closed timestamp is within 0.5s of word_time
        eye_closed = any(abs(word_time - ts) < 0.5 for ts in closed_timestamps)
        result_analysis.append({"word": word, "closed": eye_closed})
        word_time += 0.5

    return jsonify(result_analysis)

###############################################################################
# Main Entry Point
###############################################################################
if __name__ == '__main__':
    init_db()
    app.run(debug=True)