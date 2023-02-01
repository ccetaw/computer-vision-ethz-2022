# ETHz Computer Vision HS2022 

Personal notes for ETHz Computer Vision HS2022 written in obsidian flavored but also github flavored markdown. You could also find pdf version but they might not be properly adjusted since I output them directly from my obsidian vault. Using obsidian and topaz theme is mostly recommended.

The notes are mainly adapted from the course slides in a more organized way. Some topics that are not explained in details in the slides are extended in the notes. Materials and papers I used to write these notes are included in the repo and you could find the references in the notes.

You could also find the answer to the assignments. 

## [Camera Model](/notes_github//notes_github/Camera%20Model.md)

### Pinhole Camera

### Camera Calibration

## [Local Features](/notes_github//notes_github/Local%20Features.md)

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
#### [Histogram of Oriented Gradients](/notes_github/Histogram%20of%20Oriented%20Gradients.md)


## [Optical Flow](/notes_github/Optical%20Flow.md)

### Problem Setting & Definition
#### Brightness Constancy Equation
#### Small Aperture Problem

### Lucas-Kanade Optical Flow

### Horn-Schunck Optical Flow


## [Object Recognition](/notes_github/Object%20Recognition.md)

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

## [Image Segmentation](/notes_github/Image%20Segmentation.md)

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

## [Object Detection](/notes_github/Object%20Detection.md)

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

## [Object Tracking](/notes_github/Object%20Tracking.md)

### Pixel Tracking

### Template Tracking

#### Lucas-Kanade Template Tracker
#### Generalized LK Template Tracker
#### Track by Matching

### Track by Detection

### Online Learning

## [Projective Geometry](/notes_github/Projective%20Geometry.md)

### Projective Plane

#### Points and Lines
#### Conics
#### 2D Transformations

### Projective 3D Space
#### Points and Planes
#### Quadrics
#### 3D Transformations

## [Epipolar Geometry](/notes_github/Epipolar%20Geometry.md)

### The Essential Matrix

### The Fundamental Matrix
#### The Eight-Point Algorithm
#### The Normalized Eight-Point Algorithm

### Recovering Projection Matrices

## [Structure from Motion](/notes_github/Structure%20from%20Motion.md)

### Triangulation

### Affine Structure from Motion

### Perspective Structure from Motion

### Bundle Adjustment

## Stereo Vision

## [Robust Fitting](/notes_github/Robust%20Fitting.md)
### Robust Cost Functions

### RANdom SAmple Consensus - RANSAC

### Adpative RANSAC
