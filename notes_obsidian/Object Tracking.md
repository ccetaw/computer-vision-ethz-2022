---
tags:
- CV
date: 16/11/2022
---

# Tracking

## Pixel Tracking

![[Object Tracking-2.png]]

Recall that in [[Optical Flow]] we are able to calculate the velocity of all pixels in a picture. Using this velocity we could track a single point / all pixels. 

![[Object Tracking-1.png]]

## Template Tracking

[Generalized Lucas-Kanade Tracking & Kalman Filter](https://www.vision.rwth-aachen.de/media/course/SS/2016/computer-vision-2/cv2_16_exercise_01.pdf)

### Lucas-Kanade Template Tracker

We now want to track a special box instead of a single pixel
- Replace 5x5 window with user-specified template window ![[Object Tracking-3.png|400]]
- Compute flow-vector per template, not per pixel ![[Object Tracking-4.png|400]]
- Sum over pixels inside template-window
- 
$$
\begin{aligned}
E(u, v) & =\sum_{\mathbf{x}}[I(x+u, y+v)-T(x, y)]^2 \\
& \approx \sum_{\mathbf{x}}\left[I(x, y)+u I_x(x, y)+v I_y(x, y)-T(x, y)\right]^2 \\
& =\sum_{\mathbf{x}}\left[u I_x(x, y)+v I_y(x, y)+D(x, y)\right]^2 \quad \text { with } D=I-T
\end{aligned}
$$

and solve

$$
\sum_{\mathbf{x}}\left[\begin{array}{cc}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{array}\right]\left[\begin{array}{l}
u \\
v
\end{array}\right]=\sum_{\mathbf{x}}\left[\begin{array}{c}
I_x D \\
I_y D
\end{array}\right]
$$

### Generalized LK Template Tracker

![[Object Tracking-5.png|600]]

The problem with the simple Lucas-Kanade Template Tracker is that it assumes pure translation for all pixels in a larger window, which is unreasonable for long periods of time.

We now ==allow arbitrary template transformation-model instead of only pure translation==

$$
\begin{aligned}
 E(u, v)=\sum_{\mathbf{x}}&[I(x+u, y+v)-T(x, y)]^2 \\
& \Downarrow \\
E(\mathbf{p})=\sum_{\mathbf{x}}&[I(\mathbf{W}([x, y] ; \mathbf{p}))-T([x, y])]^2
\end{aligned}
$$

where $\mathbf{p}$ is the warp parameter (translation, rotation, affine, etc.)

To optimize the warped version, the Lucas-Kanade algorithm assumes that a current estimate of $\mathbf{p}$ is known and then iteratively solves for increments to the parameters $\mathbf{p}$; i.e. the following expression is (approximately) minimized:

$$
\begin{align}
\sum_{\mathbf{x}} [I(\mathbf{W}(\mathbf{x};\mathbf{p}+\Delta \mathbf{p})) - T(\mathbf{x})]^{2}
\end{align}
$$

w.r.t $\Delta \mathbf{p}$ and then the parameters are updated:

$$
\mathbf{p} \Leftarrow \mathbf{p} + \Delta \mathbf{p}
$$


The Lucas-Kanade algorithm (==which is a Gauss-Newton gradient descent non-linear optimization algorithm==), see [[Newton's Method#adaptive gradient descent|Newton's method]].

$$
\begin{aligned}
& \sum_{\mathbf{x}}[I(\mathbf{W}(\mathbf{x} ; \mathbf{p}+\Delta \mathbf{p}))-T(\mathbf{x})]^2 \\
\underset{ \text{Tayler expansion} }{ \Rightarrow } & \sum_{\mathbf{x}}\left[I(\mathbf{W}(\mathbf{x} ; \mathbf{p}))+\nabla I \frac{\partial \mathbf{W}}{\partial \mathbf{p}} \Delta \mathbf{p}-T(\mathbf{x})\right]^2 \\
\underset{ \text{Take derivative} }{ \Rightarrow } & \sum_{\mathbf{x}}\left[\nabla I \frac{\partial \mathbf{W}}{\partial \mathbf{p}}\right]^{\mathrm{T}}\left[I(\mathbf{W}(\mathbf{x} ; \mathbf{p}))+\nabla I \frac{\partial \mathbf{W}}{\partial \mathbf{p}} \Delta \mathbf{p}-T(\mathbf{x})\right] \\
\underset{ \text{Set derivative to zero} }{ \Rightarrow } & \Delta \mathbf{p}=H^{-1} \sum_{\mathbf{x}}\left[\nabla I \frac{\partial \mathbf{W}}{\partial \mathbf{p}}\right]^{\mathrm{T}}[T(\mathbf{x})-I(\mathbf{W}(\mathbf{x} ; \mathbf{p}))] \\
\text { Where } H= & \sum_x\left[\nabla I \frac{\partial \mathbf{W}}{\partial \mathbf{p}}\right]^{\mathrm{T}}\left[\nabla I \frac{\partial \mathbf{W}}{\partial \mathbf{p}}\right]
\end{aligned}
$$


> **The Lucas-Kanade Algorithm**
> Iterate:
> 	1. Warp $I$ with $\mathbf{W}(\mathbf{x};\mathbf{p})$ to compute $I(\mathbf{W}(\mathbf{x};\mathbf{p}))$
> 	2. Compute the error image $T(\mathbf{x}) - I(\mathbf{W}(\mathbf{x};\mathbf{p}))$
> 	3. Warp the gradient $\nabla I$ with $\mathbf{W}(\mathbf{x};\mathbf{p})$
> 	4. Evaluate the Jacobian $\frac{ \partial \mathbf{W} }{ \partial \mathbf{p} }$ at $(\mathbf{x};\mathbf{p})$
> 	5. Compute the steepest descent image s $\nabla I \frac{ \partial \mathbf{W} }{ \partial \mathbf{p} }$
> 	6. Compute the Hessian matrix 
> 	7. Compute $\sum_{\mathbf{x}}\left[\nabla I \frac{\partial \mathbf{W}}{\partial \mathbf{p}}\right]^{\mathrm{T}}[T(\mathbf{x})-I(\mathbf{W}(\mathbf{x} ; \mathbf{p}))]$
> 	8. Compute $\Delta \mathbf{p}$
> 	9. Update the parameters $\mathbf{p} \Leftarrow \mathbf{p} + \Delta \mathbf{p}$
> until $\|\Delta p\| \leq \epsilon$

**Pros**:
- can handle different parameter space
- can converge fast in a high-frame rate video

**Cons**:
- not robust to image noise or large displacement
- some transformations are impossible to parameterize

### Track by Matching
We already know how to [[Local Features|extract features]], i.e. interesting points or regions. We could use a template and find its interesting points and match them to every frame of the video to find it. 
![[Object Tracking-6.png|500]]

## Track by Detection

If an object is in the detection database, we could detect it in each frame independently to achieve tracking.

- Detect object(s) independently in each frame
- Associate detections over time into tracks
![[Object Tracking-7.png|600]]

## Online Learning

The idea is, we train an object-background classifier in the fly, i.e. update the classifier every frame. 
| Frame $t$                       | Frame $t+1$                |
| ------------------------------- | -------------------------- |
| ![[Object Tracking-8.png\|200]] | ![[Object Tracking-9.png\|400]] |

The tracking loop is as follows
![[Object Tracking.png]]

The problem this online learning has is that if we mask the object to be tracked gradually with another object, the tracker will end up tracking the masking object. This is called *gradual drift*.

![[Object Tracking-10.png|400]]

In the above example, the tracker will finally track the hand instead of the face. To avoid drift, we can "anchor" our model with the initial object in order to help avoid drift.


