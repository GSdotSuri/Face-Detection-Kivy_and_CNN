from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.button import Button

import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras import models


face_data = np.load("face_data.npy")
X = face_data[:, 1:]
Y = face_data[:,0]
model = models.load_model("facemodel.hdf5")


class MyLayout(GridLayout):

    def __init__(self, **args):
        super(MyLayout, self).__init__(**args)
        self.classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.capture = cv2.VideoCapture(0)
        self.cols = 2
        self.imgWidget = Image()
        self.add_widget(self.imgWidget)
        self.faceWidget = Image()
        self.add_widget(self.faceWidget)
        lbl = Label(text = "Name of the face recognized will appear here...")
        self.lbl = lbl
        self.add_widget(lbl)
        Clock.schedule_interval(self.update, 1/50)

 
    def update(self, event):
        ret,img = self.capture.read()
        if ret:
            self.img_array = img

            faces = self.classifier.detectMultiScale(img, 1.2, 5)

            if (len(faces)>=1):
                for face in faces:
                    x,y,w,h = face
                    
                    facecut = img[y:y+h, x:x+w]
                    out_facecut = cv2.resize(facecut, (300, 300))
                    gray = cv2.cvtColor(out_facecut, cv2.COLOR_BGR2GRAY)
                    grey_img = cv2.resize(gray,(300,300))
                    imgarr = np.array(grey_img).reshape((300,300,1))
                    names_dict = {1:"Kartik", 0:"Gurprasad"}
                    name = names_dict[model.predict(imgarr)[0]]
                    self.lbl.text = str(name)

                    flipped_face = facecut[::-1]
                    flipped_face_buffer = flipped_face.tobytes()
                    face_texture = Texture.create(size=(flipped_face.shape[1],flipped_face.shape[0]), colorfmt='bgr')
                    face_texture.blit_buffer(flipped_face_buffer, colorfmt='bgr', bufferfmt = 'ubyte')

                    self.faceWidget.texture = face_texture

            flipped_img = img[::-1]
            flipped_buffer = flipped_img.tobytes()
            texture = Texture.create(size=(img.shape[1],img.shape[0]), colorfmt='bgr')
            texture.blit_buffer(flipped_buffer, colorfmt='bgr', bufferfmt = 'ubyte')

            self.imgWidget.texture = texture


class MyApp(App):

    def build(self):
        return MyLayout()

app = MyApp()
app.run()