import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os

# Display
from IPython.display import Image, display
import matplotlib.cm as cm
from keras.models import load_model


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, path="cam.jpg", alpha=0.5):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(path)

    # Display Grad CAM
    display(Image(path))

def get_data_by_category(train_data, category):
    sample = train_data.loc[train_data[category] == 1]
    sample = sample[:5]
    names = sample['filename'].to_numpy()
    labels = sample['label'].to_numpy()

    return names, labels

def number_to_string(label):
    if label == 0:
        return 'CNV'
    elif label == 1:
        return 'DME'
    elif label == 2 :
        return 'DRUSEN'
    elif label == 2 :
        return 'NORMAL'
        return 'nothing'

def load_nn_model():
    vgg19 = tf.keras.applications.VGG19(include_top=False,weights='imagenet',input_tensor=None,input_shape=(150, 150, 3),pooling=None)
    vgg19.trainable = False
    model_vgg = tf.keras.models.Sequential([vgg19,tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same'), tf.keras.layers.PReLU(alpha_initializer='zeros'),  tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same'),tf.keras.layers.PReLU(alpha_initializer='zeros'),tf.keras.layers.Flatten(),tf.keras.layers.Dense(100),tf.keras.layers.PReLU(alpha_initializer='zeros'),tf.keras.layers.Dense(4, activation='softmax')])
    metrics = ['accuracy', tfa.metrics.F1Score(num_classes=4),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]
    model_vgg.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=metrics)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(ROOT_DIR + '/static/model/vgg19_model.h5')
    model_vgg.load_weights(model_path)
    return model_vgg

def visualize_hetmaps(path, last_conv_layer_name, current_time):
    model = load_nn_model()
    print("----------------------------------")
    print(model.summary())
    print("----------------------------------")
    print(model.layers[3].name)
    img_array = get_img_array(path, size=(150,150))
    model.layers[-1].activation = None
    heatmap = make_gradcam_heatmap(img_array, model, model.layers[3].name)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    gradcam_path = os.path.join(ROOT_DIR + f'/static/images/output_gradcam{current_time}.png')
    save_and_display_gradcam(img_path=path, heatmap=heatmap, path=gradcam_path)