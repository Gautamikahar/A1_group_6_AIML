import cv2
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

def create_dataset():
    student_id = entry_id.get().strip()
    name = entry_name.get().strip()

    if not student_id or not name:
        messagebox.showerror("Input Error", "Please enter both Student ID and Name.")
        return

    dataset_path = 'dataset'
    os.makedirs(dataset_path, exist_ok=True)

    # Folder name as "ID_Name"
    user_folder = os.path.join(dataset_path, f"{student_id}_{name}")
    os.makedirs(user_folder, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0

    messagebox.showinfo("Instructions", "Camera starting...\nPress 'q' to stop capturing manually.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow(f"Capturing - {name} (Press 'q' to stop)", frame)

        # Save image every few frames
        if count % 10 == 0:
            img_name = os.path.join(user_folder, f"{count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"[INFO] Saved: {img_name}")
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Dataset created for {name} (ID: {student_id}) successfully!")

# ---------------- Tkinter GUI ----------------
root = tk.Tk()
root.title("Dataset Creator - Face Attendance System")
root.geometry("400x300")
root.config(bg="#f5f5f5")

title = tk.Label(root, text="📸 Dataset Creator", font=("Arial", 18, "bold"), bg="#f5f5f5", fg="#333")
title.pack(pady=20)

frame = tk.Frame(root, bg="#f5f5f5")
frame.pack(pady=10)

tk.Label(frame, text="Student ID:", font=("Arial", 12), bg="#f5f5f5").grid(row=0, column=0, padx=10, pady=10)
entry_id = tk.Entry(frame, font=("Arial", 12))
entry_id.grid(row=0, column=1)

tk.Label(frame, text="Student Name:", font=("Arial", 12), bg="#f5f5f5").grid(row=1, column=0, padx=10, pady=10)
entry_name = tk.Entry(frame, font=("Arial", 12))
entry_name.grid(row=1, column=1)

btn_create = ttk.Button(root, text="Start Creating Dataset", command=create_dataset)
btn_create.pack(pady=20)

root.mainloop()
