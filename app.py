import face_recognition
import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, send_file, redirect, url_for, jsonify, request, session
from datetime import datetime
import csv
import pickle
from functools import wraps
import glob

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

# Hardcoded admin credentials (replace with a database in production)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"

# List of subjects (you can modify this list as needed)
SUBJECTS = ["Statistics and Integral Calculas", "Science of Nature", "Applied Mechanics", "Discrete Structures", "Essential of Data Science"]

# Load known face encodings from encodings.pickle
encoding_file = "encodings.pickle"
known_face_encodings = []
known_faces = []  # List of dictionaries: [{"name": "...", "roll_no": "...", "division": "..."}, ...]

def load_encodings():
    """Load face encodings from encodings.pickle"""
    global known_face_encodings, known_faces
    if os.path.exists(encoding_file):
        with open(encoding_file, "rb") as file:
            data = pickle.load(file)
        known_face_encodings = data["encodings"]
        
        # Check if the data uses the new structure ("faces") or the old structure ("names")
        if "faces" in data:
            known_faces = data["faces"]
        elif "names" in data:
            # Convert old structure to new structure
            known_faces = [{"name": name, "roll_no": "N/A", "division": "N/A"} for name in data["names"]]
            # Update the encodings.pickle file to the new structure
            updated_data = {"encodings": known_face_encodings, "faces": known_faces}
            with open(encoding_file, "wb") as file:
                pickle.dump(updated_data, file)
        else:
            known_faces = []
    else:
        known_face_encodings = []
        known_faces = []

# Load encodings initially
load_encodings()

# Dictionary to track last attendance time
last_attendance = {}
video_capture = None  # Global video capture object
stop_feed = False  # Flag to stop the feed
last_marked = {}  # To store the last marked attendance details
attendance_marked = False  # Flag to indicate if attendance was marked
capture_face = False  # Flag to indicate if face capture is in progress
new_person = {}  # Store the details of the new person being added: {"name": "...", "roll_no": "...", "division": "..."}
face_captured = False  # Flag to indicate if a face was successfully captured

# Decorator to require login for admin routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def mark_attendance(name, subject):
    """Mark attendance for recognized person"""
    global last_marked, attendance_marked
    current_time = datetime.now()
    
    if name in last_attendance:
        time_diff = (current_time - last_attendance[name]).total_seconds()
        if time_diff < 300:  # 5-minute cooldown
            return False
    
    if not os.path.exists('attendance'):
        os.makedirs('attendance')
    
    date_str = current_time.strftime('%Y-%m-%d')
    # Create a subject-specific filename
    filename = f'attendance/attendance_{subject.lower()}_{date_str}.csv'
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time'])
        writer.writerow([name, current_time.strftime('%m/%d/%Y'), 
                        current_time.strftime('%H:%M:%S')])
    
    last_marked = {
        'name': name,
        'subject': subject,
        'date': current_time.strftime('%d-%m-%y'),
        'time': current_time.strftime('%H:%M:%S'),
        'filename': filename
    }
    
    last_attendance[name] = current_time
    attendance_marked = True
    return True

def capture_new_face():
    """Capture a new face for adding to encodings"""
    global video_capture, capture_face, new_person, face_captured
    capture_face = True
    face_captured = False
    
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
    
    while capture_face:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            known_face_encodings.append(face_encoding)
            known_faces.append(new_person)  # Add the new person's details
            
            # Save the updated encodings
            data = {"encodings": known_face_encodings, "faces": known_faces}
            with open(encoding_file, "wb") as file:
                pickle.dump(data, file)
            
            face_captured = True
            capture_face = False
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    video_capture.release()
    video_capture = None

def recognize_faces(subject):
    global video_capture, stop_feed, attendance_marked
    stop_feed = False
    attendance_marked = False
    
    # Reload encodings to ensure we have the latest data
    load_encodings()
    
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
    
    while not stop_feed:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            status = ""

            if matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_faces[best_match_index]["name"]  # Extract name from the dictionary
                    if mark_attendance(name, subject):  # Pass the subject to mark_attendance
                        status = "Attendance Marked"
                        stop_feed = True
                    else:
                        status = "Already Marked"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} - {status}", 
                       (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    video_capture.release()
    video_capture = None

@app.route('/')
def index():
    return render_template('index.html', subjects=SUBJECTS)

@app.route('/select_subject', methods=['POST'])
def select_subject():
    subject = request.form['subject']
    session['subject'] = subject
    return redirect(url_for('camera'))

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('admin_panel'))
        else:
            return render_template('admin_login.html', error=True)
    return render_template('admin_login.html', error=False)

@app.route('/admin_panel')
@login_required
def admin_panel():
    # Get all attendance files and organize them by subject
    attendance_files = {}
    for subject in SUBJECTS:
        # Find all CSV files for this subject
        files = glob.glob(f'attendance/attendance_{subject.lower()}_*.csv')
        # Normalize paths for the current OS
        files = [os.path.normpath(file) for file in files]
        attendance_files[subject] = sorted(files, key=os.path.getmtime, reverse=True)  # Sort by modification time (newest first)
    
    # Debug: Print the attendance files
    print("Attendance files:", attendance_files)
    
    return render_template('admin_panel.html', known_faces=known_faces, attendance_files=attendance_files)

@app.route('/add_face', methods=['GET', 'POST'])
@login_required
def add_face():
    global new_person
    if request.method == 'POST':
        new_person = {
            "name": request.form['name'],
            "roll_no": request.form['roll_no'],
            "division": request.form['division']
        }
        return redirect(url_for('capture_face'))
    return render_template('add_face.html')

@app.route('/capture_face')
@login_required
def capture_face():
    return render_template('capture_face.html', new_person=new_person)

@app.route('/capture_feed')
@login_required
def capture_feed():
    return Response(capture_new_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_capture_status')
@login_required
def check_capture_status():
    global face_captured
    return jsonify({'captured': face_captured})

@app.route('/face_added')
@login_required
def face_added():
    global face_captured, new_person
    face_captured = False  # Reset the flag
    return render_template('face_added.html', new_person=new_person)

@app.route('/delete_face/<name>', methods=['POST'])
@login_required
def delete_face(name):
    global known_face_encodings, known_faces
    # Reload encodings to ensure we have the latest data
    load_encodings()
    
    # Find the index of the person with the given name
    index_to_delete = None
    for i, person in enumerate(known_faces):
        if person["name"] == name:
            index_to_delete = i
            break
    
    if index_to_delete is not None:
        known_faces.pop(index_to_delete)
        known_face_encodings.pop(index_to_delete)
        
        # Save the updated encodings
        data = {"encodings": known_face_encodings, "faces": known_faces}
        with open(encoding_file, "wb") as file:
            pickle.dump(data, file)
    
    return redirect(url_for('admin_panel'))

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    session.pop('subject', None)  # Clear the subject from the session
    return redirect(url_for('index'))

@app.route('/confirmation')
def confirmation():
    global attendance_marked
    if not last_marked:
        return redirect(url_for('index'))
    
    attendance_marked = False
    return render_template('confirmation.html', last_marked=last_marked)

@app.route('/download/<path:filename>')
def download(filename):
    # Debug: Print the filename being requested
    print("Requested filename:", filename)
    
    # Normalize the path for the current OS
    filename = os.path.normpath(filename)
    
    # Ensure the filename is within the attendance directory to prevent directory traversal
    if not filename.startswith('attendance' + os.sep):
        print("Filename does not start with 'attendance/'. Redirecting to index.")
        return redirect(url_for('index'))
    
    # Debug: Check if the file exists
    if not os.path.exists(filename):
        print(f"File does not exist: {filename}. Redirecting to index.")
        return redirect(url_for('index'))
    
    # Debug: File exists, proceeding to send
    print(f"File exists: {filename}. Sending file for download.")
    # Set the download name to the basename of the file
    download_name = os.path.basename(filename)
    return send_file(filename, as_attachment=True, download_name=download_name)

@app.route('/check_status')
def check_status():
    global attendance_marked
    return jsonify({'marked': attendance_marked})

@app.route('/video_feed')
def video_feed():
    # Retrieve the subject from the session within the request context
    subject = session.get('subject', 'Unknown')
    return Response(recognize_faces(subject), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    if 'subject' not in session:
        return redirect(url_for('index'))
    return render_template('camera.html')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if video_capture is not None:
            video_capture.release()