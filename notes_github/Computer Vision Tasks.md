---
tags:
- CV
date: 22/01/2022
---

# Computer Vision Tasks

![Computer Vision Tasks](attachments/Computer%20Vision%20Tasks.png)



## Object Recognition

Object recognition is the technique of identifying the object present in images and videos. It is one of the most important applications of machine learning and deep learning. The goal of this field is to teach machines to understand (recognize) the content of an image just like humans do. ==**Object detection is a subset of object recognition**==

## Image Classification

==In Image classification, it takes an image as an input and outputs the classification label of that image with some metric (probability, loss, accuracy, etc).== For Example: An image of a cat can be classified as a class label “cat” or an image of Dog can be classified as a class label “dog” with some probability.

![Computer Vision Tasks](attachments/Computer%20Vision%20Tasks.jpg)

## Object Localization

Object Localization algorithm ==locates the presence of an object in the image and represents it with a bounding box.== It takes an image as input and outputs the location of the bounding box in the form of (position, height, and width).

## Object Detection

==Object Detection algorithms act as a combination of image classification and object localization.== It takes an image as input and produces one or more bounding boxes with the class label attached to each bounding box. These algorithms are capable enough to deal with multi-class classification and localization as well as to deal with the objects with multiple occurrences. ^e80a35

![Computer Vision Tasks-1](attachments/Computer%20Vision%20Tasks-1.png)

**Challenges of Object Detection**
-   In object detection, the bounding boxes are always rectangular. So, it does not help with determining the shape of objects if the object contains the curvature part.
-   Object detection cannot accurately estimate some measurements such as the area of an object, perimeter of an object from image.

## Image Segmentation

==Image segmentation is a further extension of object detection in which we mark the presence of an object through pixel-wise masks generated for each object in the image==. This technique is more granular than bounding box generation because this can helps us in determining the shape of each object present in the image. This granularity helps us in various fields such as medical image processing, satellite imaging, etc.

![Computer Vision Tasks-2](attachments/Computer%20Vision%20Tasks-2.png)

There are primarily two types of segmentation:

-   **Instance Segmentation:** Identifying the boundaries of the object and label their pixel with different colors.
-   **Semantic Segmentation:** Labeling each pixel in the image (including background) with different colors based on their category class or class label.

![Computer Vision Tasks-3](attachments/Computer%20Vision%20Tasks-3.png)
