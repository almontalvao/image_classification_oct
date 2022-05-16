import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from.gradcam import visualize_hetmaps
from PIL import Image


def read_image(image):
    return mpimg.imread(image)


def format_image(image):
   image = tf.io.decode_image(image, dtype=tf.float32, channels=3)
   converted = image[tf.newaxis, ...]
   resized = tf.image.resize(converted, [150, 150])
   return  resized



def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/model/')
    tflite_model_file = 'oct_converted_model.tflite'

    #input_img = read_image(img)
    format_img = format_image(img)

    interpreter = tf.lite.Interpreter(model_path=path+tflite_model_file)
    interpreter.allocate_tensors()
    print(interpreter.get_input_details())
    input_details = interpreter.get_input_details()[0]["index"]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details, format_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    prediction = output.argmax()

    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    return class_names[prediction]


def plot_category(img, current_time):
    """Plot the input image

    Args:
        img [jpg]: image file
    """
    format_img = Image.open(img)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(ROOT_DIR + f'/static/images/output_{current_time}.png')

    if os.path.exists(file_path):
        os.remove(file_path)

    plt.imsave(file_path, format_img)
    visualize_hetmaps(file_path, 'conv2d_1', current_time)
