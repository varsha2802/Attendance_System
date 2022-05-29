
from flask import Flask, render_template, Response, request, url_for, redirect, flash, abort, send_from_directory
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from datetime import date
import time
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///todo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = "False"
db = SQLAlchemy(app)


class Todo(db.Model):
    EnrollmentNo = db.Column(db.Integer, primary_key="True")
    name = db.Column(db.String(20), nullable="False")
    department = db.Column(db.String(40), nullable="False")
    date_created = db.Column(db.DateTime, default=datetime.utcnow)


    def __repr__(self) -> str:
        return f"{self.EnrollmemtNo} - {self.name} - {self.department} "


from app import db

db.create_all()
@app.route("/", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        enrollmentNo = request.form['EnrollmentNo']
        name = request.form['name']
        department = request.form['department']
        todo = Todo(EnrollmentNo=enrollmentNo, name=name, department=department)
        db.session.add(todo)
        db.session.commit()
    allTodo = Todo.query.all()
    return render_template('index.html', allTodo=allTodo)


path = 'ImagesAttendance'
images = []
classnames = []
myList= os.listdir(path)
print(myList)
for cl in myList:
    #curImg =cv2.imread(f'{path}/{cl}')
    curImg=cv2.imread(os.path.join('ImagesAttendance',cl))
    if curImg is None:
        continue
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def find_encodings(images):
    encodelist= []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


encodelistknown= find_encodings(images)
print('Encoding complete')

def markatt(name):
    with open('attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist =[]
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])

        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('5H:%M:%S')
            today=date.today()
            f.writelines(f'\n{name}, {dtstring},{today}')


def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture_duration = 10
    start_time = time.time()
    while (int(time.time() - start_time) < capture_duration):
        success, img = cap.read()
        # read the camera frame
        if not success:
            break
        else:
            imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

            facescur = face_recognition.face_locations(imgs)
            encodecur = face_recognition.face_encodings(imgs, facescur)

            for encodeFace, faceloc in zip(encodecur, facescur):
                matches = face_recognition.compare_faces(encodelistknown, encodeFace)
                facedis = face_recognition.face_distance(encodelistknown, encodeFace)
                matchindex = np.argmin(facedis)

                if matches[matchindex]:
                    name = classnames[matchindex].upper()
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    markatt(name)
                    print('Encoding complete')

            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

            #cv2.imshow('webcam', img)
            #cv2.waitKey(10)
'''
def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture_duration = 10
    start_time = time.time()
    while (int(time.time() - start_time) < capture_duration):
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(encodelistknown, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(encodelistknown, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = encodelistknown[best_match_index]

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("window", frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

'''



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
