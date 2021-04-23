
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox as messagebok
from tkinter.filedialog import askopenfilename


app = tk.Tk()
app.geometry('200x200')
app['bg'] = '#0059b3'
app.title("Clasificador de recipientes")

modelo = './modelo/structureModelo.h5'
layerWeights = './modelo/layerWeights.h5'

labels = ["Aluminum","Cardboard","Ceramics","Crystal","Mud","Natural","Paper","Plastic","Unicel","Wood"]

def Training(file):
  model = load_model(modelo)
  model.load_weights(layerWeights)
  x = load_img(file, target_size=(120, 120))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  print(result)
  answer = np.argmax(result)
  print(labels[answer])
  label = tk.Label(app, text=labels[answer], font="none 12 bold")
  label.grid(row=3, column=2, sticky="n")
  return labels[answer]


def extractImage ():
  '''
  filename = askopenfilename()
  print(filename)
  if(filename!=""):
    '''
  Training('Aluminio.jpg')

button = tk.Button(app, text="Analizar imágen", command= extractImage)
button.grid(row=0, column=0)



app.mainloop()

