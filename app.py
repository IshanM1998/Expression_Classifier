#|export

from fastai.vision.all import *
import gradio as gr

#|export
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


#|export
learn = load_learner('resnet18_emotion_detection1.pkl')

#|export

categories = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

#|export
image = gr.inputs.Image(shape=(256,256))
label = gr.outputs.Label()
examples = ['angry.jpg','disgust.jpg', 'fear.jpg', 'happy.jpg', 'neutral.jpg', 'sad.jpg', 'surprise.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch()