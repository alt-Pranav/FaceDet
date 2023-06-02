import streamlit as st
import cv2
from mtcnn import MTCNN 

# comment
detector = MTCNN()

def detect_bounding_box(vid):
    # detect faces in the image
    faces = detector.detect_faces(vid)
    #for face in faces:
    #   print(face) 
    
    for face in faces:
        x, y, width, height = face['box']
        conf = face['confidence']
        cv2.rectangle(vid, (x, y), (x + width, y + height), (0, 255, 0), 4)
        cv2.putText(vid, str(conf)[:4], (x+width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return faces

def main():
    st.title("Face Detection App")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_bounding_box(frame)  # apply the function we created to the video frame

        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')


if __name__ == "__main__":
    main()