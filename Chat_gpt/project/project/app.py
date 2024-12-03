from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition
import os
import numpy as np
import datetime

app = Flask(__name__)

# Directory to save registered faces
REGISTERED_FACES_DIR = "static/registered_faces"
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)

# Directory to save attendance records
ATTENDANCE_FILE = "attendance.csv"

# Helper function to save face encodings
def save_face_encoding(image_path, name):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        encoding_path = os.path.join(REGISTERED_FACES_DIR, f"{name}.npy")
        np.save(encoding_path, face_encodings[0])
        return True
    return False

# Helper function to log attendance
def mark_attendance(name):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a") as file:
        file.write(f"{name},{date},{time}\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        if name:
            # Capture 50 frames from the webcam
            cap = cv2.VideoCapture(0)
            count = 0
            success = False
            while count < 50:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("Registering Face", frame)
                    # Save the first frame as the image for registration
                    if count == 0:
                        image_path = os.path.join(REGISTERED_FACES_DIR, f"{name}.jpg")
                        cv2.imwrite(image_path, frame)
                    count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

            # Save face encoding after capturing frames
            if save_face_encoding(image_path, name):
                success = True

            if success:
                return "Face registered successfully!"
            else:
                os.remove(image_path)
                return "Face registration failed. Ensure the image contains a single face."
    return render_template('register.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    # Open webcam for real-time recognition
    cap = cv2.VideoCapture(0)
    recognized_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            for face_file in os.listdir(REGISTERED_FACES_DIR):
                if face_file.endswith(".npy"):
                    name = os.path.splitext(face_file)[0]
                    registered_encoding = np.load(os.path.join(REGISTERED_FACES_DIR, face_file))
                    result = face_recognition.compare_faces([registered_encoding], face_encoding, tolerance=0.6)
                    if result[0]:
                        recognized_names.add(name)
                        mark_attendance(name)
                        cv2.putText(frame, f"Hello, {name}!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        break

        # Display the frame with annotations
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return f"Recognized: {', '.join(recognized_names)}"

if __name__ == '__main__':
    app.run(debug=True)
