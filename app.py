from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime, time
import pandas as pd
import os
import time as t
import threading

app = Flask(__name__)
absent_marked = {}
frame_count_dict = {}  # To track consecutive frames per person

# -------------------------------
# Load encodings
# -------------------------------
with open("encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# -------------------------------
# Attendance File Setup
# -------------------------------
ATTENDANCE_FILE = "attendance.csv"
if os.path.exists(ATTENDANCE_FILE):
    attendance = pd.read_csv(ATTENDANCE_FILE)
else:
    attendance = pd.DataFrame(columns=["Name", "Date", "Subject", "In_Time", "Out_Time", "Status"])
    attendance.to_csv(ATTENDANCE_FILE, index=False)

# -------------------------------
# Globals
# -------------------------------
camera = None
camera_running = False
process_this_frame = True

# -------------------------------
# Lecture slots
# -------------------------------
LECTURE_SLOTS = [
    {"subject": "Maths", "start": "09:00", "end": "10:00"},
    {"subject": "DSA", "start": "10:00", "end": "11:00"},
    {"subject": "OS", "start": "11:00", "end": "12:00"},
    {"subject": "BREAK", "start": "12:00", "end": "13:00"},
    {"subject": "CN", "start": "13:00", "end": "14:00"},
    {"subject": "AI", "start": "14:00", "end": "15:00"},
    {"subject": "DBMS", "start": "15:00", "end": "16:00"},
]

GRACE_MINUTES = 10
MIN_FRAME_COUNT = 3  # Minimum consecutive frames to mark attendance

# -------------------------------
# Get current slot
# -------------------------------
def get_current_slot():
    now = datetime.now().time()
    for slot in LECTURE_SLOTS:
        if slot["subject"] == "BREAK":
            continue
        start = datetime.strptime(slot["start"], "%H:%M").time()
        end = datetime.strptime(slot["end"], "%H:%M").time()
        if start <= now <= end:
            return slot
    return None

# -------------------------------
# Mark attendance
# -------------------------------
def mark_attendance(name):
    global attendance
    attendance = pd.read_csv(ATTENDANCE_FILE)

    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    now_obj = datetime.now().time()

    slot = get_current_slot()
    if slot is None:
        return

    subject = slot["subject"]
    start_time = datetime.strptime(slot["start"], "%H:%M")
    grace_limit = (start_time + pd.Timedelta(minutes=GRACE_MINUTES)).time()

    # Remove duplicates first
    attendance = attendance.drop_duplicates(subset=["Name", "Date", "Subject"], keep="first")
    attendance.to_csv(ATTENDANCE_FILE, index=False)

    mask = (attendance["Name"] == name) & (attendance["Date"] == today) & (attendance["Subject"] == subject)

    if not mask.any():
        status = "Present" if now_obj <= grace_limit else "Late"
        new_entry = pd.DataFrame([[name, today, subject, now_time, "", status]],
                                 columns=["Name", "Date", "Subject", "In_Time", "Out_Time", "Status"])
        attendance = pd.concat([attendance, new_entry], ignore_index=True)
        attendance.to_csv(ATTENDANCE_FILE, index=False)
        return

    idx = attendance[mask].index[0]
    attendance.at[idx, "Out_Time"] = now_time
    attendance.to_csv(ATTENDANCE_FILE, index=False)

# -------------------------------
# Background Absent Checker
# -------------------------------
def background_absent_checker():
    while True:
        check_absent_and_early_leave()
        t.sleep(60)

# -------------------------------
# Absent / Early leave logic
# -------------------------------
def check_absent_and_early_leave():
    global attendance, absent_marked
    if not os.path.exists(ATTENDANCE_FILE):
        return

    attendance = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().time()

    for slot in LECTURE_SLOTS:
        if slot["subject"] == "BREAK":
            continue

        subject = slot["subject"]
        end_time = datetime.strptime(slot["end"], "%H:%M").time()
        if now > end_time:
            lecture_key = f"{today}_{subject}"
            if absent_marked.get(lecture_key, False):
                continue

            for s in known_names:
                exists = attendance[(attendance["Name"] == s) &
                                    (attendance["Date"] == today) &
                                    (attendance["Subject"] == subject)]
                if exists.empty:
                    new_entry = pd.DataFrame([[s, today, subject, "", "", "Absent"]],
                                             columns=["Name", "Date", "Subject", "In_Time", "Out_Time", "Status"])
                    attendance = pd.concat([attendance, new_entry], ignore_index=True)

            # Early leave detection
            for idx, row in attendance[(attendance["Date"] == today) & (attendance["Subject"] == subject)].iterrows():
                if row["Out_Time"] != "" and row["Out_Time"] < slot["end"]:
                    attendance.at[idx, "Status"] = "Early Leave"

            attendance.to_csv(ATTENDANCE_FILE, index=False)
            absent_marked[lecture_key] = True

# -------------------------------
# Video Frame Generator
# -------------------------------
def generate_frames():
    global camera, camera_running, process_this_frame, frame_count_dict
    while camera_running:
        success, frame = camera.read()
        if not success:
            continue

        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        face_names = []

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small)
            face_locations = [f for f in face_locations if (f[2]-f[0])>50 and (f[1]-f[3])>50]  # filter small faces
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for enc in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.45)
                name = "Unknown Face"

                if True in matches:
                    idx = matches.index(True)
                    name = known_names[idx]
                    frame_count_dict[name] = frame_count_dict.get(name, 0) + 1

                    # Only mark if detected in MIN_FRAME_COUNT consecutive frames
                    if frame_count_dict[name] == MIN_FRAME_COUNT:
                        mark_attendance(name)
                face_names.append((name, (0,255,0) if name != "Unknown Face" else (0,0,255)))

        process_this_frame = not process_this_frame

        # Draw results
        for (top,right,bottom,left), (name,color) in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
            cv2.rectangle(frame, (left, top-35), (right, top), color, -1)
            cv2.putText(frame, name, (left+8, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        t.sleep(0.01)

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, camera_running
    if not camera_running:
        camera = cv2.VideoCapture(0)
        camera_running = True
        return jsonify({"status":"started"})
    return jsonify({"status":"already_running"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, camera_running
    if camera_running:
        camera_running = False
        camera.release()
        return jsonify({"status":"stopped"})
    return jsonify({"status":"already_stopped"})

@app.route('/video')
def video():
    if not camera_running:
        return "Camera not started", 400
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance_page():
    attendance = pd.read_csv(ATTENDANCE_FILE)
    attendance["Date"] = pd.to_datetime(attendance["Date"], errors='coerce').dt.date
    unique_dates = sorted(list(attendance["Date"].dropna().unique()), reverse=True)
    unique_dates_str = [d.strftime("%Y-%m-%d") for d in unique_dates]

    selected_date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    selected_date_obj = datetime.strptime(selected_date, "%Y-%m-%d").date()

    filtered = attendance[attendance["Date"] == selected_date_obj]
    total = len(filtered)
    table_html = filtered.to_html(classes="data", index=False)

    return render_template("attendance.html",
                           table=table_html,
                           total=total,
                           dates=unique_dates_str,
                           selected_date=selected_date)

# -------------------------------
if __name__ == "__main__":
    threading.Thread(target=background_absent_checker, daemon=True).start()
    app.run(debug=True)
