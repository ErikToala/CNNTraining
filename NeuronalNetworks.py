import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

modelo = './modelo/structureModelo.h5'
layerWeights = './modelo/layerWeights.h5'
model = load_model(modelo)
model.load_weights(layerWeights)
labels = ["Aluminum","Cardboard","Ceramics","Crystal","Mud","Natural","Paper","Plastic","Unicel","Wood"]
#labels = ["Crystal","Cardboard","Ceramics","Aluminum","Wood","Natural","Paper","Plastic","Unicel","Mud"]

def Training(file):
  x = load_img(file, target_size=(320, 320))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  print(result)
  answer = np.argmax(result)
  if answer == 0:
    print(labels[0])
  elif answer == 1:
    print(labels[1])
  elif answer == 2:
    print(labels[2])
  elif answer == 3:
    print(labels[3])
  elif answer == 4:
    print(labels[4])
  elif answer == 5:
    print(labels[5])
  elif answer == 6:
    print(labels[6])
  elif answer == 7:
    print(labels[7])
  elif answer == 8:
    print(labels[8])
  elif answer == 9:
    print(labels[9])

  return answer


Training('natural01.jpg')