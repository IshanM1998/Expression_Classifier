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

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image_in = gr.inputs.Image(shape=(256,256))
label = gr.outputs.Label()
examples = ['angry.jpg','disgust.jpg', 'fear.jpg', 'happy.jpg', 'neutral.jpg', 'sad.jpg', 'surprise.jpg']

# img = np.array(image_in)
gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

img_pil = Image.fromarray(gray)
img_pil = img_pil.resize((48,48))

intf = gr.Interface(fn=classify_image, inputs=img_pil, outputs=label, examples=examples)
intf.launch()