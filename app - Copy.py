from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
import time

app = Flask(__name__)
DATASET_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Halaman utama
@app.route("/")
def index():
    return render_template("index.html")

# Halaman login wajah
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(0)
        matched_user = None
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y : y + h, x : x + w]
                face_img = cv2.resize(face_img, (200, 200))

                for filename in os.listdir(MODEL_DIR):
                    if not filename.endswith(".yml"):
                        continue
                    username = os.path.splitext(filename)[0]
                    model_path = os.path.join(MODEL_DIR, filename)

                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(model_path)
                    label, confidence = recognizer.predict(face_img)

                    if confidence < 55:
                        matched_user = username
                        break
                if matched_user:
                    break

            if matched_user:
                break

            cv2.imshow("Login Wajah", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Stop after 5 detik kalau gak cocok
            if time.time() - start_time > 5:
                break

        cap.release()
        cv2.destroyAllWindows()

        if matched_user:
            return redirect(url_for("dashboard", username=matched_user))
        else:
            return render_template("gagal.html")
    return render_template("login.html")


# Halaman dashboard
@app.route("/dashboard")
def dashboard():
    username = request.args.get("username")
    return render_template("dashboard.html", username=username)

# Halaman registrasi user baru
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        user_folder = os.path.join(DATASET_DIR, username)
        os.makedirs(user_folder, exist_ok=True)

        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        count = 0

        while count < 30:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                file_path = os.path.join(user_folder, f"{count}.jpg")
                cv2.imwrite(file_path, face)
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Scan Wajah", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Train model
        images, labels = [], []
        for file in os.listdir(user_folder):
            img = cv2.imread(os.path.join(user_folder, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(0)

        if images:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(images, np.array(labels))
            model.save(os.path.join(MODEL_DIR, f"{username}.yml"))

        return redirect(url_for("login"))

    return render_template("register.html")


if __name__ == "__main__":
    app.run(debug=True)
