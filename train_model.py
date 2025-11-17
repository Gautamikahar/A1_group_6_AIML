import os
import cv2
import face_recognition
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
RF_MODEL_FILE = "rf_model.pkl"
XGB_MODEL_FILE = "xgb_model.pkl"

known_encodings = []
known_names = []

print("\n🚀 Training started...\n")

# -------------------------------
# Extract Encodings
# -------------------------------
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"📂 Processing: {person_name}")
    person_encodings = []

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")

        if len(boxes) != 1:
            continue

        encoding = face_recognition.face_encodings(rgb, boxes)[0]
        person_encodings.append(encoding)

    if person_encodings:
        known_encodings.extend(person_encodings)
        known_names.extend([person_name] * len(person_encodings))
        print(f"✅ Encoded {len(person_encodings)} images for {person_name}")
    else:
        print(f"⚠️ No valid faces found for {person_name}")

# -------------------------------
# SAVE ENCODINGS
# -------------------------------
if known_encodings:
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_encodings, known_names), f)
    print(f"\n💾 Encodings saved to {ENCODINGS_FILE}")
else:
    print("\n❌ No encodings found. Exiting.")
    exit()

# -------------------------------
# Convert to numpy arrays
# -------------------------------
X = np.array(known_encodings)
y = np.array(known_names)

# -------------------------------
# Label Encoding for XGBoost
# -------------------------------
print("\n🔢 Encoding labels for XGBoost...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save encoder
pickle.dump(le, open("label_encoder.pkl", "wb"))
print("💾 Label Encoder saved as label_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -------------------------------
# Train Random Forest
# -------------------------------
print("\n🌳 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

with open(RF_MODEL_FILE, "wb") as f:
    pickle.dump(rf, f)

print(f"✅ RF accuracy: {rf_acc * 100:.2f}%")
print(f"💾 Random Forest saved as {RF_MODEL_FILE}")

# -------------------------------
# Train XGBoost
# -------------------------------
print("\n🔥 Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    eval_metric='mlogloss'
)

xgb.fit(X_train_enc, y_train_enc)
xgb_pred = xgb.predict(X_test_enc)
xgb_acc = accuracy_score(y_test_enc, xgb_pred)

with open(XGB_MODEL_FILE, "wb") as f:
    pickle.dump(xgb, f)

print(f"✅ XGBoost accuracy: {xgb_acc * 100:.2f}%")
print(f"💾 XGBoost saved as {XGB_MODEL_FILE}")
