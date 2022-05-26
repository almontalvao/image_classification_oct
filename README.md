# retina-oct-images
Fourthbrain Capstone project.

We use supervised and self-supervised deep learning techniques to classifiy OCT images. The data set is available at: 

```python
https://www.kaggle.com/datasets/paultimothymooney/kermany2018
```

We developed a prediction API, using Flask and Elastic Beanstalk, which allows us to upload retina images and it generates a prediction. Grad-CAM is also used to highlight the most important areas of the image prediction. The API is available here:

```python
http://ec2-3-21-231-50.us-east-2.compute.amazonaws.com:5000/
```

