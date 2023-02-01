---
tags:
- CV
date: 22-09-2022
---

# Projective Geometry

[Projective Geometry: A Short Introduction](http://morpheo.inrialpes.fr/people/Boyer/Teaching/M2R/geoProj.pdf)

## Geometry in 3D and images
### 3D -> 2D conversion 
![[PG_1.png|400]]

- Line-preserving: straight lines(3D) -> straight lines(2D)
- Size of objects: inverse proportional to distance
- No angle-preserving
- Vanishing point:
	- Parallel lines are not parallel anymore
	- All mapped parallel lines intersect in a vanishing point
	- Vanishing point at infinity


## The Projective Plane

![[PG_2.png]]

The projective space $\mathcal{P}^{2}$, associated to the vector space $\mathbb{R}^{3}$, is called the projective plane. Its importance in visual computing domains is coming from the fact that the image plane of a 3D world projection can be seen as a projective plane and that relationships between images of the same 3D scene can be modeled through projective transformations.

### Points and Lines

A point in $\mathcal{P}^{2}$ is represented by $3$ homogeneous coordinates $(x_{0},x_{1},x_{2})^{\top}$ defined up to scale factor.  

A line in $\mathcal{P}^{2}$ can also be represented by $3$ homogeneous coordinates $(l_{0},l_{1},l_{2})^{\top}$: consider $2$ points $A$ and $B$ of $\mathcal{P}^{2}$ and the line going through them. A third point $C$ belongs to this line only if the coordinates of $A$, $B$ and $C$ are linearly dependent, i.e. their determinant vanishes

$$
\left|\begin{array}{ccc}
x_0^A & x_0^B & x_0^C \\
x_1^A & x_1^B & x_1^C \\
x_2^A & x_2^B & x_2^C
\end{array}\right|=0
$$

which could be rewritten as 

$$
l_{0}x_{0}^{C} + l_{1}x_{1}^{C} + l_{2}x_{2}^{C} = (l_{0},l_{1},l_{2}) \cdot \begin{pmatrix}
x_{0}^{C} \\
x_{1}^{C} \\
x_{2}^{C}
\end{pmatrix} = L^{\top}\cdot \mathbf{x}^{C} = 0
$$

where the $l_{i}$s are functions of the coordinates of $A$ and $B$:

$$
l_0=\left|\begin{array}{ll}
x_1^A & x_1^B \\
x_2^A & x_2^B
\end{array}\right|,\quad l_1=-\left|\begin{array}{ll}
x_0^A & x_0^B \\
x_2^A & x_2^B
\end{array}\right|, \quad l_2=\left|\begin{array}{cc}
x_0^A & x_0^B \\
x_1^A & x_1^B
\end{array}\right| .
$$


#### DUality points-lines

It could be easily seen that the line joining $A$ and $B$ is (by definition of cross product)

$$
L = \mathbf{x}^{A} \times \mathbf{x}^{B}
$$


For a point $\mathbf{x}$, if $L_{1}$ and $L_{2}$ both pass it, we have

$$
\begin{cases}
L_{1}^{\top} \cdot \mathbf{x} = 0 \\
L_{2}^{\top} \cdot \mathbf{x} = 0
\end{cases}
$$

thus 

$$
\mathbf{x} = L_{1} \times L_{2}
$$


#### Points and lines at infinity

Points such that $x_{2}=0$ define a hyperplane of $\mathcal{P}^{2}$ called the line at infinity. The line at infinity contains all points at infinity. 

In homogeneous coordinates, the line at infinity is $(0,0,1)$.

![[Projective Geometry.png]]

### 2D Transformations
![[PG_3.png]]

1. Scaling, DoF  = 2

$$
\begin{bmatrix}
a & 0 \\
0 & b
\end{bmatrix}
$$


2. Shearing, DoF  = 1

$$
\begin{bmatrix}
1 & a \\
b & 1 
\end{bmatrix}
$$


3. Rotation, DoF  = 1

$$
\begin{bmatrix}
\cos \theta & -\sin \theta  \\
\sin \theta & \cos \theta 
\end{bmatrix}
$$


4. Translation, DoF  = 2
No possible matrix representation in 2D.

$$
\begin{bmatrix}
1 & 0 & t_{x} \\
0 & 1 & t_{y}  \\
0 & 0 & 1
\end{bmatrix}
$$


5. Euclidean(rigid) = rotation + translation, DoF  = 3

$$
\begin{bmatrix}
r_{1} & r_{2} & r_{3}  \\
r_{4} & r_{5} & r_{6} \\
0 & 0 & 1
\end{bmatrix}
= 
\begin{bmatrix}
\cos\theta & -\sin\theta & r_{3} \\
\sin\theta & \cos\theta & r_{6}  \\
0 & 0 & 1
\end{bmatrix}
$$


6. Similarity = scaling + rotation + translation, DoF  = 4

$$
\begin{bmatrix}
r_{1} & r_{2} & r_{3}  \\
r_{4} & r_{5} & r_{6} \\
0 & 0 & 1
\end{bmatrix}
= 
\begin{bmatrix}
s\cos\theta & -s\sin\theta & r_{3} \\
s\sin\theta & s\cos\theta & r_{6}  \\
0 & 0 & 1
\end{bmatrix}
$$

(rotation part multiplied by scale $s$).

7. Affine = scaling + shearing + rotation + translation, DoF  = 6

$$
\begin{bmatrix}
a_{1} & a_{2} & a_{3} \\
a_{4} & a_{5} & a_{6} \\
0 & 0 & 1
\end{bmatrix}
$$

8. Projective(last elements specified by scale), DoF  = 8

$$
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
$$



### Conics
A conic is a planar curve described by a second degree homogeneous (defined up to a scale factor) equation

$$
ax_{0}^{2}+bx_{0}x_{1}+cx_{1}^{2}+dx_{0}+ex_{1}+f = 0
$$

where $(x_{0},x_{1})$ are the affine coordinates in the plane. In homogeneous coordinates, replace $x_{0}$, $x_{1}$ with $\frac{x_{0}}{x_{2}}$ and $\frac{x_{1}}{x_{2}}$ respectively.

Using matrix notation

$$
\mathbf{x}^{\top } \mathbf{C}\mathbf{x}
$$

where

$$
\mathbf{C} = \begin{bmatrix}
a  & b/2 & d/2 \\
b/2 & c & e/2  \\
d/2 & e/2 & f
\end{bmatrix}
$$

is called the homogeneous matrix associated to the conic. 

There are $5$ degrees of freedom: $\{ a,b,c,d,e,f \}$ (conic defined up to scale) thus five points define a conic. For each point the conic passes though

$$
ax^{2}+bxy+cy^{2}+dx+ey+f=0
$$

or 

$$
(x^{2},xy,y^{2},x,y,1)\mathbf{c} = 0
$$

stack five constraints yields 

$$
\left[\begin{array}{llllll}
x_1^2 & x_1 y_1 & y_1^2 & x_1 & y_1 & 1 \\
x_2^2 & x_2 y_2 & y_2^2 & x_2 & y_2 & 1 \\
x_3^2 & x_3 y_3 & y_3^2 & x_3 & y_3 & 1 \\
x_4^2 & x_4 y_4 & y_4^2 & x_4 & y_4 & 1 \\
x_5^2 & x_5 y_5 & y_5^2 & x_5 & y_5 & 1
\end{array}\right] \mathbf{c}=0
$$


#### Dual conics
The line $L$ tangent to $\mathbf{C}$ at point $\mathbf{x}$ on $\mathbf{C}$ is given by $L=\mathbf{C}\mathbf{x}$.

> **Proof**:
> The line $L=\mathbf{C}\mathbf{x}$ is going through $\mathbf{x}$ since $L^{\top}\mathbf{x} = \mathbf{x}^{\top}\mathbf{C}\mathbf{x}=0$. Assume that another point $\mathbf{y}$ of $L$ also belongs to $\mathbf{C}$ then $\mathbf{y}^{\top}\mathbf{C}\mathbf{y}=0$ and $\mathbf{x}^{\top}\mathbf{C}\mathbf{y} = 0$ and hence any point $\mathbf{x}+k\mathbf{y}$ along the line defined by $\mathbf{x}$ and $\mathbf{y}$ belongs to $\mathbf{C}$ as well since
> 
> $$(\mathbf{x}+k\mathbf{y})^{\top}\mathbf{C}(\mathbf{x}+k\mathbf{y}) = 0$$
> 
> Thus $L$ goes through $\mathbf{x}$ and is tangent to $\mathbf{C}$

The set of lines $L$ tangent to $\mathbf{C}$ satisfies the equation
$$
L^{\top} \mathbf{C}^{*} L = 0
$$
In general $\mathbf{C}^{*} = \mathbf{C}^{-1}$. 

> **Proof**:
> Simply replace $L$ with $\mathbf{C}\mathbf{x}$ we prove the equation

$C^{*}$ is the dual conic of $\mathbf{C}$ or the conic envelop
![[Projective Geometry-1.png|400]]

#### Degenerate Conics
When the matrix $\mathbf{C}$ is singular the associated conic is said to be degenerated.

Example:
2 lines $L_{1}$ and $L_{2}$ define a degenerate conic $\mathbf{C} = L_{1}\cdot L_{2}^{\top} +L_{2}\cdot L_{1}^{\top}$ 

### Projective transformations
A projectivity is an invertible mapping $h$ form $\mathcal{P}^{2}$ to itself such that three points $\mathbf{x}_{1},\mathbf{x}_{2},\mathbf{x}_{3}$ lie on the same line if and only if $h(\mathbf{x}_{1}), h(\mathbf{x}_{2}), h(\mathbf{x}_{3})$ lie on the same line.

> Theorem:
> A mapping $h: \mathcal{P}^{2}\to \mathcal{P}^{2}$ is a projectivity if and only if there exist a *non-singular* $3\times 3$ matrix $H$ such that for any point in $\mathcal{P}^{2}$ represented by a vector $\mathbf{x}$ it is true that $h(\mathbf{x}) = H\mathbf{x}$.

The projective transformation is thus

$$
\mathbf{x}' = H\mathbf{x} 
$$

where 

$$
H = \begin{bmatrix}
h_{11} &  h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31}  & h_{32} & h_{33}
\end{bmatrix}
$$

which is of 8 degrees of freedom (instead of 9 up to a scale factor).

| Transf. group | Dof | Matrix                                                                                                                                         | Deformation                         | Invariants                                                                  |
| ------------- | --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------- |
| Euclidean     | 3   | $$\left[\begin{array}{ccc}\cos \theta & -\sin \theta &T_0\\\sin \theta & \cos \theta & T_1 \\0 & 0 & 1\end{array}\right]$$                     | ![[Projective Geometry-2.png\|150]] | length, area                                                                |
| Isometry      | 4   | $$\left[\begin{array}{ccc}\epsilon \cos \theta & -\sin \theta & T_0 \\\epsilon \sin \theta & \cos \theta & T_1 \\0 & 0 & 1\end{array}\right]$$ | ![[Projective Geometry-3.png\|150]] | length ratio, angle                                                         |
| Affine        | 6   | $$\left[\begin{array}{ccc}a_1 & a_2 & a_3 \\a_4 & a_5 & a_6 \\0 & 0 & 1\end{array}\right]$$                                                    | ![[Projective Geometry-4.png\|150]] | parallelism, area ratio, length ratio on a line, linear vector combinations |
| Projective    | 8   | $$\begin{bmatrix}h_{11} &  h_{12} & h_{13} \\h_{21} & h_{22} & h_{23} \\h_{31}  & h_{32} & h_{33}\end{bmatrix}$$                               | ![[Projective Geometry-5.png\|150]] | incidence, collinearity, concurrence, cross-ratio                                                                            |

To determine a projective transformation given points before and after transformations, we need $4$ points for an exact solution for $H$. Since

$$
\lambda\left[\begin{array}{c}
x^{\prime} \\
y^{\prime} \\
1
\end{array}\right]=\left[\begin{array}{lll}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{array}\right]\left[\begin{array}{c}
x \\
y \\
1
\end{array}\right]
$$

we have $2$ independent equations for one point. As we have $8$ Dof, we need at least 4 points to determine $H$. If more points are observed, we have not exact solution, because measurements are inexact (noise in measurement).

**Transformation of 2D points, lines and conics**
- Points: $\mathbf{x}' = H \mathbf{x}$
- Lines: $L' = H^{-\top}L$
- Conics: $\mathbf{C}' = H^{-\top}\mathbf{C}H^{-1}$
- Dual conics: $\mathbf{C}'^{*} = H\mathbf{C}^{*}H^{\top}$

**Fixed points and lines**
![[Projective Geometry-8.png|500]]
- Eigenvectors of $H$ are fixed points
- Eigenvectors of $H^{-\top}$ are fixed lines

**Line at infinity**
The line at infinity $L_{\infty}$ is a fixed line under a projective transformation $H$ if and only if $H=H_{A}$ is an affinity.
$$
L'_{\infty} = H_{A}^{-\top}L_{\infty} = \begin{bmatrix}
A^{-\top} & 0 \\
-\mathbf{t}A^{-\top} & 1
\end{bmatrix}
\begin{pmatrix}
0 \\
0 \\
1
\end{pmatrix}
= L_{\infty}
$$

### Circular Points and Its Conic Dual
Two *circular points* are defined to be 

$$
\begin{align}
I &= (1,i, 0)^{\top} \\
J  & = (1,-i,0)^{\top}
\end{align}
$$

which satisfies

$$
x_{1}^{2}+x_{2}^{2} = 0
$$

Circular points algebraically codes orthogonal directions

$$
I = (1,0,0)^{\top} + i(0,1,0)^{\top}
$$

The conic dual to the circular points is 

$$
\mathbf{C}_{\infty}^{*} = IJ^{\top} + JI^{\top} = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

The dual conic $\mathbf{C}_{\infty}^{*}$ is fixed conic under the projective transformation $H$ if and only if $H=H_{S}$ is a similarity.

$$
\mathbf{C}_{\infty}^{*} = H_{S}\mathbf{C}_{\infty}^{*}H_{S}^{\top}
$$

### Angles
Let $l(l_{1},l_{2},l_{3})$ and $m(m_{1},m_{2},m_{3})$ be two lines, and they form an angle $\theta$. Their Euclidean angle is 

$$
\cos\theta = \frac{l_{1}m_{1}+l_{2}m_{2}}{\sqrt{ (l_{1}^{2}+l_{2}^{2})(m_{1}^{2}+m_{2}^{2}) }}
$$

The projective angle is 

$$
\cos\theta = \frac{l^{\top}\mathbf{C}_{\infty}^{*}m}{\sqrt{ (l^{\top}\mathbf{C}_{\infty}^{*}l)(m^{\top}\mathbf{C}_{\infty}^{*}m) }}
$$


### Direct Linear Transformation(DLT)
Let $\mathbf{x}_{i}' = H\mathbf{x}_{i}$, then

$$
\mathbf{x}_{i}' \times H \mathbf{x}_{i} = 0
$$

Denote $\mathbf{x}_{i}' = (x_{i}', y_{i}', w_{i}')$ and 

$$
H = \begin{pmatrix}
\mathbf{h}_{1}^{\top} \\
\mathbf{h}_{2}^{\top} \\
\mathbf{h}_{3}^{\top}
\end{pmatrix}
$$

Then 

$$
H\mathbf{x}_{i} = \begin{pmatrix}
\mathbf{h}_{1}^{\top}\mathbf{x}_{i} \\
\mathbf{h}_{2}^{\top}\mathbf{x}_{i} \\
\mathbf{h}_{3}^{\top}\mathbf{x}_{i}
\end{pmatrix}
$$

and 

$$
\mathbf{x}_{i}' \times H\mathbf{x}_{i} = \begin{pmatrix}
y_{i}' \mathbf{h}_{3}^{\top}\mathbf{x}_{i} - w_{i}'\mathbf{h}_{2}^{\top}\mathbf{x}_{i} \\
w_{i}' \mathbf{h}_{1}^{\top}\mathbf{x}_{i} - x_{i}'\mathbf{h}_{3}^{\top}\mathbf{x}_{i} \\
x_{i}' \mathbf{h}_{2}^{\top}\mathbf{x}_{i} - y_{i}'\mathbf{h}_{1}^{\top}\mathbf{x}_{i}
\end{pmatrix}
$$

which is equivalent to 

$$
\begin{bmatrix}
\mathbf{0}^{\top}  & -w_{i}'\mathbf{x}_{i}^{\top} & y_{i}'\mathbf{x}_{i}^{\top} \\
w_{i}'\mathbf{x}_{i}\top & \mathbf{0}^{\top} & -x_{i}'\mathbf{x}_{i}^{\top} \\
-y_{i}'\mathbf{x}_{i}^{\top} & x_{i}'\mathbf{x}_{i}^{\top} & \mathbf{0}^{\top}
\end{bmatrix}
\begin{pmatrix}
\mathbf{h}_{1} \\
\mathbf{h}_{2} \\
\mathbf{h}_{3}
\end{pmatrix} = \mathbf{0}
$$

Since only 2 out 3 equations are linearly independent, we could drop the third row(only if $w_{i}' \ne 0$).

Denote 

$$
\begin{bmatrix}
\mathbf{0}^{\top}  & -w_{i}'\mathbf{x}_{i}^{\top} & y_{i}'\mathbf{x}_{i}^{\top} \\
w_{i}'\mathbf{x}_{i}\top & \mathbf{0}^{\top} & -x_{i}'\mathbf{x}_{i}^{\top} 
\end{bmatrix} = A_{i}
$$

stack $A_{i}$s to get 

$$
A = \begin{bmatrix}
A_{1} \\
A_{2} \\
A_{3} \\
A_{4}
\end{bmatrix}
$$

and solve $A\mathbf{h} = 0$ for $H$, where

$$
\mathbf{h} = \begin{pmatrix}
\mathbf{h}_{1} \\
\mathbf{h}_{2} \\
\mathbf{h}_{3}
\end{pmatrix}
$$

$A$ could be $8\times9$ or $12\times9$, but of rank $8$. 

If we have more than $4$ point pairs, we get an overdetermined equation. No exact solution exist because of inexact measurement. To find approximate solution, we could 
- Add additional constraint needed to avoid $0$, e.g. $\|\mathbf{h}\| = 1$
- If $A\mathbf{h} = 0$ not possible, try to minimize $\|A\mathbf{h}\|$

> **DLT algorithm**
> Objective:
> 	Given $n\geq 4$ 2D to 2D point correspondences $\{ \mathbf{x}_{i} \leftrightarrow \mathbf{x}_{i}'\}$ , determine the 2D homography matrix $H$ such that $\mathbf{x}_{i}' = H\mathbf{x}_{i}$
> Algorithm:
> 	1. For each correspondence $\mathbf{x}_{i} \leftrightarrow \mathbf{x}_{i}'$ compute $A_{i}$. Usually only two first rows needed.
> 	2. Assemble $n$ $2\times 9$ matrices $A_{i}$ into a single $2n\times 9$ matrix $A$.
> 	3. Obtain [[SVD(Singular Value Decomposition)|SVD]] of $A$. Solution for $\mathbf{h}$ is the last column of $V$.
> 	4. Determine $H$ from $\mathbf{h}$.

Previously we say if $A\mathbf{h}=0$ is not possible, we try to minimize $\|A\mathbf{h}\|$. There are several ways we can minimize the term, in different definitions of cost functions.

#### Algebraic Distance
Define:
- $\mathbf{e} = A\mathbf{h}$, the residual vector
- $\mathbf{e}_{i} = A_{i}\mathbf{h}$, the partial vector for each $\mathbf{x}_{i} \leftrightarrow \mathbf{x}_{i}'$
- $d_{\text{alg}}^{2}(\mathbf{x}_{1},\mathbf{x}_{2}) = a_{1}^{2}+a_{2}^{2}$ where $\mathbf{a} = (a_{1},a_{2},a_{3})^{\top} = \mathbf{x}_{1}\times \mathbf{x}_{2}$

Thus 

$$
d_{\text{alg}}^{2}(\mathbf{x}_{i}', H\mathbf{x}_{i}) = \|\mathbf{e}_{i}\|^{2} = \left\|\left[\begin{array}{ccc}
0^{\top} & -w_i^{\prime} \mathbf{x}_i^{\top} & y_i^{\prime} \mathbf{x}_i^{\top} \\
w_i^{\prime} \mathbf{x}_i^{\top} & 0^{\top} & -x_i^{\prime} \mathbf{x}_i^{\top}
\end{array}\right] \mathbf{h}\right\|^2
$$

and 

$$
\sum_{i} d_{\text{alg}}^{2}(\mathbf{x}_{i}',H\mathbf{x}_{i}) = \sum_{i}\|\mathbf{e}_{i}\|^{2} = \|A\mathbf{h}\|^{2} = \|\mathbf{e}\|^{2}
$$

Algebraic distance is not geometrically/statistically meaningful, but given good normalization it works fine and is very fast (use for initialization).

It could be easily seen that DLT minimizes $\|A\mathbf{h}\|$. 

#### Geometric Distance
Define:
- $\mathbf{x}$, measured coordinates
- $\hat{\mathbf{x}}$, estimated coordinates
- $\overline{\mathbf{x}}$, true coordinates
- $d(\cdot,\cdot)$, Euclidean distance (in an image)

Then define 
- Error in one image 

$$
\hat{H} = \underset{ H }{ \arg \min } \sum_{i} d^{2}(\mathbf{x}_{i}', H\overline{\mathbf{x}}_{i})
$$

- Symmetric transfer error

$$
\hat{H} = \underset{ H }{ \arg\min } \sum_{i}[d^{2}(\mathbf{x}_{i}, H^{-1}\mathbf{x}_{i}') + d^{2}(\mathbf{x}_{i}', H\mathbf{x}_{i})]
$$

![[Projective Geometry-6.png|600]]
- Reprojection error

$$
\begin{array}{r}
\left(\hat{H}, \hat{\mathbf{x}}_i, \hat{\mathbf{x}}_i^{\prime}\right)=\underset{H, \hat{\mathbf{x}}_i, \hat{\mathbf{x}}_i^{\prime}}{\arg\min} \sum_i d\left(\mathbf{x}_i, \hat{\mathbf{x}}_{\mathrm{i}}\right)^2+d\left(\mathbf{x}_i^{\prime}, \hat{\mathbf{x}}^{\prime}\right)^2 \\
\text { subject to } \hat{\mathbf{x}}_i^{\prime}=\hat{H} \hat{\mathbf{x}}_i
\end{array}
$$

![[Projective Geometry-7.png|600]]

#### Comparison of Geometric and Algebraic Distances
Denote $\mathbf{x}_{i}' = (x_{i}', y_{i}', w_{i}')$, $\hat{\mathbf{x}} = (\hat{x}_{i}', \hat{y}_{i}', \hat{w}_{i}') = H\overline{\mathbf{x}}$. And we have (you might need to deduce this)

$$
A_{i}\mathbf{h} = \mathbf{e}_{i} = \begin{pmatrix}
y_{i}' \hat{w}_{i}' - w_{i}'\hat{y}_{i}' \\
w_{i}'\hat{x}_{i}' - x_{i}' \hat{w}_{i}'
\end{pmatrix}
$$

Then the algebraic distance is 

$$
d^{2}_{\text{alg}}(\mathbf{x}_{i}', \hat{\mathbf{x}}_{i}') = \left(y_i^{\prime} \hat{w}_i^{\prime}-w_i^{\prime} \hat{y}_i^{\prime}\right)^2+\left(w_i^{\prime} \hat{x}_i^{\prime}-x_i^{\prime} \hat{w}_i^{\prime}\right)^2
$$


and the Euclidean distance is 

$$
d^{2}(\mathbf{x}_{i}', \hat{\mathbf{x}}_{i}') = \left(y_i^{\prime} / w_i^{\prime}-\hat{y}_i^{\prime} / \hat{w}_i^{\prime}\right)^2+\left(\hat{x}_i^{\prime} / \hat{w}_i^{\prime}-x_i^{\prime} / w_i^{\prime}\right)^2 = \frac{d^{2}_{\text{alg}}(\mathbf{x}_{i}', \hat{\mathbf{x}}_{i}')}{(w_{i}'\hat{w}_{i}')^{2}}
$$

$w_{i}' = 1$ typically, and $\hat{w}_{i}' = \mathbf{x}_{i}^{\top}\mathbf{h}_{3}$, but for affinities $\hat{w}_{i}'=1$, too.  Thus for affinities DLT can minimize geometric distance.

#### Statistical Cost FUnction and MLE
Assume zero-mean isotropic Gaussian noise

$$
P(x) = \frac{1}{2\pi\sigma^{2}} e^{ -d^{2}(\mathbf{x}, \overline{\mathbf{x}})/(2\sigma^{2}) }
$$

The pdf of error in one image is 

$$
P(\{ \mathbf{x}_{i}' \}|H) = \prod_{i} \frac{1}{2\pi\sigma^{2}} e^{ -d^{2}(\mathbf{x}_{i}', H\overline{\mathbf{x}}_{i})/(2\sigma^{2}) }
$$

The log-likelihood is then

$$
\text{LL} = -\frac{1}{2\sigma^{2}}\sum_{i} d^{2}(\mathbf{x}_{i}', H\overline{\mathbf{x}}_{i}) + \text{const}
$$

MLE is thus equivalent to minimization of geometric distance.

The pdf of error in both images is 

$$
P(\{ \mathbf{x}_{i}' \}|H) = \prod_{i} \frac{1}{2\pi\sigma^{2}} e^{ -(d\left(\mathbf{x}_i, \hat{\mathbf{x}}_{\mathrm{i}}\right)^2+d\left(\mathbf{x}_i^{\prime}, \hat{\mathbf{x}}^{\prime}\right)^2)/(2\sigma^{2}) }
$$

MLE is equivalent to the minimization of reprojection error.

##### Mahalonobis distance
This is the general Gaussian case. Observations are now not independent. Measurement $X$ with covariance matrix $\Sigma$.

$$
\|X-\overline{X}\|_{\Sigma}^{2} = (X-\overline{X})^{\top}\Sigma^{-1}(X-\overline{X})
$$

## Projective 3D Space
### Points and Planes
The homogeneous representation of points in $\mathcal{P}^{3}$ is 

$$
\mathbf{x} = \begin{pmatrix}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4}
\end{pmatrix}
$$

and a plane is represented by 

$$
\pi = \begin{pmatrix}
\pi_{1} \\
\pi_{2} \\
\pi_{3} \\
\pi_{4}
\end{pmatrix}
$$

The point $\mathbf{x}$ lies on the plane if and only if $\pi^{\top}\mathbf{x} = 0$, and the plane $\pi$ goes through the point $\mathbf{x}$ if and only if $\pi^{\top}\mathbf{x}=0$.

#### Determine a Plane from Points
Three points determine a plane

$$
\begin{bmatrix}
\mathbf{x}_{1}^{\top} \\
\mathbf{x}_{2}^{\top} \\
\mathbf{x}_{3}^{\top}
\end{bmatrix} \pi = 0
$$

#### Determine a Point from Planes
Three planes determine a point

$$
\begin{bmatrix}
\pi_{1}^{\top} \\
\pi_{2}^{\top} \\
\pi_{3}^{\top}
\end{bmatrix}
\mathbf{x} = 0
$$


Let $\{ \mathbf{x}_{1},\mathbf{x}_{2},\mathbf{x}_{3} \}$ be a span of a plane $\pi$, then

$$
\pi^{\top}[\mathbf{x}_{1},\mathbf{x}_{2},\mathbf{x}_{3}] = 0
$$

### Lines
Lines are represented by its span:
![[Projective Geometry-9.png|300]]

$$
W = \begin{bmatrix}
A^{\top} \\
B^{\top}
\end{bmatrix}, \quad \lambda A+\mu B
$$

The dual representation of a line is 

$$
W^{*} = \begin{bmatrix}
P^{\top} \\
Q^{\top}
\end{bmatrix}, \quad \lambda P+\mu Q
$$

Example: $X$-axis

$$
W = \begin{bmatrix}
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0
\end{bmatrix}
\quad
W^{*} = \begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}
$$


### Points, Lines and Planes
#### A line and a Point => A plane

$$
M = \begin{bmatrix}
W \\
\mathbf{x}^{\top}
\end{bmatrix}
\quad
M\pi = 0
$$

![[Projective Geometry-10.png|300]]

#### A Line and a Plane => A Point

$$
M = \begin{bmatrix}
W^{*} \\
\pi^{\top}
\end{bmatrix}
\quad
M\mathbf{x} = 0
$$

![[Projective Geometry-11.png|300]]

### Quadrics and Dual Quadrics
A quadric $Q$ satisfies

$$
\mathbf{x}^{\top}Q\mathbf{x} = 0
$$

where $Q$ is a $4\times 4$ symmetric matrix.

- 9 Dof, i.e. 9 points define a quadric
- tangent plane of a quadric at $\mathbf{x}$ is $\pi = Q\mathbf{x}$

The dual quadric is $Q^{*}$ and in general $Q^{*} = Q^{-1}$, and it satisfies

$$
\pi^{\top}Q^{*}\pi = 0
$$

### Transformation of 3D Points, Planes and Quadrics
- Points: $\mathbf{x}' = H \mathbf{x}$
- Planes: $\pi' = H^{-\top}\pi$
- Quadrics: $Q' = H^{-\top}QH^{-1}$
- Dual conics: $Q'^{*} = HQ^{*}H^{\top}$


| Trans.     | Dof | Matrix                                                                    | Deformation                     | Invariants                                                     |
| ---------- | --- | ------------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------------------- |
| Euclidean  | 6   | $$\begin{bmatrix}R & \mathbf{t}  \\\mathbf{0}^{\top}  & 1 \end{bmatrix}$$ | ![[Projective Geometry-15.png]] | Volume                                                         |
| Similarity | 7   | $$\begin{bmatrix}sR & \mathbf{t}  \\\mathbf{0}^{\top} & 1\end{bmatrix}$$  | ![[Projective Geometry-14.png]] | Angles, ratios of length. The absolute conic $\Omega_{\infty}$ | 
| Affine     | 12  | $$\begin{bmatrix}A  & \mathbf{t} \\\mathbf{0}^{\top} & 1\end{bmatrix}$$   | ![[Projective Geometry-13.png]] |   Parallelism of planes, volume ratios, centroids, the plane at infinity $\pi_{\infty}$                                                             |
| Projective | 15  | $$\begin{bmatrix}A & \mathbf{t}  \\\mathbf{v}^{\top} &v\end{bmatrix}$$    | ![[Projective Geometry-12.png]] |  Intersection and tagency                                                              |

**The Plane at Infinity**
The plane at infinity $\pi_{\infty}$ is a fixed plane under a projective transformation $H$ if and only if $H = H_{A}$ is an affinity.

$$
\pi_{\infty}' = H_{A}^{-\top}\pi_{\infty} = \begin{bmatrix}
A^{-\top} & \mathbf{0} \\
-\mathbf{t}^{\top}A^{-\top} & 1
\end{bmatrix}
\begin{pmatrix}
0 \\
0 \\
0 \\
1
\end{pmatrix}
= \pi_{\infty}
$$

- canonical position $\pi_{\infty} = (0,0,0,1)^{\top}$
- contains directions $D = (x_{1},x_{2},x_{3},0)^{\top}$
- two planes are parallel $\iff$ line of intersection in $\pi_{\infty}$
- parallel lines $\iff$ point of intersection in $\pi_{\infty}$

### The Absolute Conic
The absolute conic $\Omega_{\infty}$ is a (point) conic on $\pi_{\infty}$. In metric frame 

$$
\begin{cases}
x_{1}^{2}+x_{2}^{2}+x_{3}^{2}  & = 0 \\
x_{4} & =0
\end{cases}
$$

The absolute conic $\Omega_{\infty}$ is a fixed conic under the projective transformation $H$ if and only is $H$ is a similarity. 

The absolute dual quadric is 

$$
\Omega_{\infty}^{*} = \begin{bmatrix}
I  &  \mathbf{0} \\
\mathbf{0}^{\top} & 0 
\end{bmatrix}
$$

The absolute conic $\Omega_{\infty}^{*}$ is a fixed conic under the projective transformation $H$ if and only is $H$ is a similarity. 

### Angles

$$
\cos\theta = \frac{\pi_{1}^{\top}\Omega_{\infty}^{*}\pi_{2}}{\sqrt{ (\pi_{1}^{\top}\Omega_{\infty}^{*}\pi_{1})(\pi_{2}^{\top}\Omega_{\infty }^{*}\pi_{2}) }}
$$


### Action of Projective Camera on Points and Lines
Denote $\mathbf{x}_{w}$ world coordinates and $\mathbf{x}_{p}$ the picture coordinates.

#### Projection of Points

$$
\mathbf{x}_{p} = P\mathbf{x}_{w} = P\begin{bmatrix}
R^{\top} & -R^{\top}\mathbf{t}  \\
\mathbf{0}^{\top}  & 1
\end{bmatrix}
\begin{bmatrix}
R & \mathbf{t}  \\
\mathbf{0}^{\top} & 1
\end{bmatrix}
\mathbf{x}_{w}
$$

#### Forward Projection of Lines
Lines => Lines

$$
P(A + \mu B) = PA + \mu PB = \mathbf{a} + \mu \mathbf{b}
$$

#### Back-projection of lines
Lines => Planes

$$\pi = P^{\top}l$$

**Proof**:

$$
\pi^{\top}\mathbf{x}_{w} = l^{\top}P\mathbf{x}_{w} = l^{\top}\mathbf{x}_{p}
$$

$l^{\top}\mathbf{x}_{p}=0$ as long as $\pi^{\top}\mathbf{x}_{w}=0$, i.e. points on plane $\pi$ after projection must be on line $l$.

### Action of projective camera on conics and quadrics
#### Back-projection to COne
![[Projective Geometry-18.png]]

$$
Q = P^{\top}\mathbf{C}P
$$
**Proof**:

$$
\mathbf{x}_{p}^{\top}\mathbf{C}\mathbf{x}_{p} = \mathbf{x}_{w}^{\top}P^{\top}\mathbf{C}P\mathbf{x}_{w} = \mathbf{x}_{w}^{\top}Q\mathbf{x}_{w}
$$


#### Forward-Projection to Quadric
![[Projective Geometry-19.png|250]]

$$
\mathbf{C}^{*} = PQ^{*}P^{\top}
$$
**Proof**:

$$
\pi^{\top}Q^{*}\pi = l^{\top}PQ^{*}P^{\top}l = l^{\top}C^{*}l
$$
