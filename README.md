# Automatic Face Recognition Attendance System

![Project Banner](static/cummins.jpg)

## Overview
This project implements an **automatic attendance system** using **face recognition**. It captures live video from a webcam, identifies students in real-time, and marks attendance in a CSV file. The system also tracks late arrivals, early leave, and absent students automatically.

This system is suitable for classrooms, training centers, or labs to automate attendance without manual intervention.

---

## Features

- ✅ Real-time face recognition using **OpenCV** and **face_recognition** library
- ✅ Automatic attendance marking in a **CSV file**
- ✅ Tracks **Late**, **Early Leave**, and **Absent** students
- ✅ Division/section selection before starting the camera
- ✅ Live video feed with detected faces highlighted
- ✅ Beautiful UI with **college-themed background**
- ✅ Background thread for **absent checking** without freezing the video feed
- ✅ Day-wise attendance view with totals

---

## AI/ML Models Used

- **CNN (Convolutional Neural Network)**: Extracts facial features from images.
- **KNN (K-Nearest Neighbors)**: Classifier used to identify faces based on embeddings.
- **Face Recognition Library**: Pre-trained models for encoding faces (based on deep learning).
- **Optional Extensions**: Other classifiers can be plugged in (SVM, Logistic Regression) for experimentation.
- Random Forest (RF): Optional ensemble classifier for student identification or prediction.
-XGBoost: Gradient boosting classifier used optionally for higher accuracy in identifying faces.



These models work together to ensure accurate recognition and reduce false positives.

---

## Technologies Used

- Python 3.9+
- Flask (Web Framework)
- OpenCV (Video capture and face detection)
- face_recognition (Facial recognition)
- Pandas (Attendance CSV management)
- HTML, CSS, JavaScript (Front-end)
- Threading (Background absent checker)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Face-Attendance-System.git
   cd Face-Attendance-System
