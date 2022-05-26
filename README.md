# retina-oct-images
Fourthbrain Capstone project.

We use supervised and self-supervised deep learning techniques to classifiy OCT images. The data set is available at: 

```python
https://www.kaggle.com/datasets/paultimothymooney/kermany2018
```

The models were developed in TensorFlow and Keras. We also developed a prediction API, using Flask and Elastic Beanstalk, which allows us to upload retina images and it generates a label prediction. Grad-CAM is used to highlight the most important areas for the prediction. The API is available here:

```python
http://ec2-3-21-231-50.us-east-2.compute.amazonaws.com:5000/
```

