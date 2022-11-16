#|export

from fastai.vision.all import *
import gradio as gr
'''
Modules for face_location and image manipulation
'''
from PIL import Image
import numpy as np
import cv2

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


learn = load_learner('resnet18_emotion_detection1.pkl')


categories = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

def classify_image(img_in):
    img = np.array(img_in)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # Detect faces
    f = 1.05
    faces =()
    # Detect faces
    while len(faces)<1 and f>1.01:
        f*= 0.97
        if f<1:
            f = 1.01
        faces = face_cascade.detectMultiScale(gray, f, 1)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = gray[y:y + h , x:x +w]
        
    # Convert cv2 image, which is an array to a PIL image format for ease of use    
    if len(faces) > 0:
        img_pil = Image.fromarray(faces)
    else:
        img_pil = Image.fromarray(gray)

    # img_pil.thumbnail((48,48))
    img_pil = img_pil.resize((48,48))
    img_arr = np.array(img_pil) 

    pred, idx, probs = learn.predict(PILImage.create(img_arr))
    return dict(zip(categories, map(float, probs)))

image_in = gr.inputs.Image(type='pil')
label = gr.outputs.Label()
examples = ['angry.jpg','disgust.jpg', 'fear.jpg', 'happy.jpg', 'neutral.jpg', 'sad.jpg', 'surprise.jpg']

intf = gr.Interface(fn=classify_image, inputs=image_in, outputs=label, examples=examples)
intf.launch()