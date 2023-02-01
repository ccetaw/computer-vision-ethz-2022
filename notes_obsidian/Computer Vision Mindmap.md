---
tags:
- CV
- Mindmap
date: 26/02/2023
---

# Computer Vision HS2022

## [[Camera Model]]

### Pinhole Camera

### Camera Calibration

## [[Local Features]]

### Keypoint Localization

#### Harris Detector Formulation

### Scale Invariant Region Selection
#### Automatic Scale Selection
##### Signature Function
- Laplacian of Gaussian
- Difference of Gaussian

### Local Descriptors
#### SIFT
#### Rotation Invariant Descriptors
#### [[Histogram of Oriented Gradients]]


## [[Optical Flow]]

### Problem Setting & Definition
#### Brightness Constancy Equation
#### Small Aperture Problem

### Lucas-Kanade Optical Flow

### Horn-Schunck Optical Flow


## [[Object Recognition]]

### Image Classification
#### Bag of Words
##### Feature Extraction
- Regular Grid
- Interesting Point Detector
- Others
##### Dictionary Learning
- K-means Clustering
##### Encode
- Quantization
- Histogram, TF-IDF
##### Classify

### Specific Object Recognition
#### Fast Lookup: Inverted Index
#### Vocabulary Tree


### Object Category Detection
- Specify Object Model
- Generate Hypotheses/Proposals
- Score Hypotheses
	- Precision
	- Recall
- Resolve Detections
	- Non-maximum Suppression
	- Context / Reasoning

## [[Image Segmentation]]

### Segmentation as Grouping
#### K-Means Clustering
#### Mixture of Gaussian
- Expectation Maximization Algorithm
#### MEan-Shift

### Hough Transform(Edge Based)

### Interactive Segmentation with GraphCuts

### Learning-based Approach
#### Traditional Machine Learning
- KNN
- Random Forests
#### Deep Learning Methods

## [[Object Detection]]

### Sliding Window Detection

### Viola/Jones Face Detector
- Image Integral and Feature Extraction
- AdaBoost Learning
- Cascade Classifier


### Implicit Shape Model

### Deep Learning Methods
#### RCNN
#### Fast RCNN
#### Faster RCNN
#### YOLO

## [[Object Tracking]]

### Pixel Tracking

### Template Tracking

#### Lucas-Kanade Template Tracker
#### Generalized LK Template Tracker
#### Track by Matching

### Track by Detection

### Online Learning

## [[Projective Geometry]]

### Projective Plane

#### Points and Lines
#### Conics
#### 2D Transformations

### Projective 3D Space
#### Points and Planes
#### Quadrics
#### 3D Transformations

## [[Epipolar Geometry]]

### The Essential Matrix

### The Fundamental Matrix
#### The Eight-Point Algorithm
#### The Normalized Eight-Point Algorithm

### Recovering Projection Matrices

## [[Structure from Motion]]

### Triangulation

### Affine Structure from Motion

### Perspective Structure from Motion

### Bundle Adjustment

## Stereo Vision

## [[Robust Fitting]]
### Robust Cost Functions

### RANdom SAmple Consensus - RANSAC

### Adpative RANSAC
