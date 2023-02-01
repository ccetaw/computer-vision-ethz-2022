# Computer Vision Exercise 5 Report
*Name*: Jingyu Wang
*Legi*: 21-956-453

## Implementation
### Color histogram
Simply call `np.histogram()` and concatenate the result.

### Derive Matrix $A$
#### No mootion at all
$A$ is simply an identity matrix since there is no motion.
$$
A = 
\begin{pmatrix}
1 & 0  \\
0 & 1 
\end{pmatrix}
$$


#### Constant velocity motion model
The position in the next moment is (assuming $dt = 1$)
$$
x + v_{x}
$$
by the first order approximation, thus 
$$
A = 
\begin{pmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1  \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

### Propagation
Generate random Gaussian noise by calling `np.random.normal()` and do a simple matrix multiplication. 

### Observation
For each particle we calculate the histogram using the function specified above and after that we calculate the weight by using the formula given in the description of the assignment.

### Estimation
Simple element-wise matrix multiplication.

### Resampling
Use `np.random.choice()` to select elements following the particle weights.

## Experiment

### Video 1

|                    | Experiment 1                      | Experiment 2                                                                             | Experiment 3                                  |
| ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------- |
| `hist_bin`         | 16                                | 16                                                                                       | 16                                            |
| `alpha`            | 0                                 | 0                                                                                        | 0                                             |
| `sigma_observe`    | 0.1                               | 0.1                                                                                      | 0.1                                           |
| `model`            | 0                                 | 0                                                                                        | 1                                             |
| `num_particles`    | 200                               | 200                                                                                      | 300                                           |
| `sigma_position`   | 15                                | 10                                                                                       | 15                                            |
| `sigma_velocity`   | 1                                 | 5                                                                                        | 1                                             |
| `initial_velocity` | (1, 10)                           | (1,10)                                                                                   | (2,-5)                                        |
| result             | ![[report.png]]                   | ![[report-1.png]]                                                                        | ![[report-2.png]]                             |
| remark             | Already able to give good results | Increase `sigma_velocity` will improve the result by being able to track the fast motion | model 1 highly depends on the intial velocity | 


### Video 2

|                    | Experiment 1                                                                             | Experiment 2                                         | Experiment 3                 |
| ------------------ | ---------------------------------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------- |
| `hist_bin`         | 16                                                                                       | 16                                                   | 16                           |
| `alpha`            | 0                                                                                        | 0                                                    | 0.8                          |
| `sigma_observe`    | 0.1                                                                                      | 0.1                                                  | 0.1                          |
| `model`            | 0                                                                                        | 1                                                    | 1                            |
| `num_particles`    | 300                                                                                      | 300                                                  | 300                          |
| `sigma_position`   | 15                                                                                       | 15                                                   | 10                           |
| `sigma_velocity`   | 1                                                                                        | 1                                                    | 1                            |
| `initial_velocity` | (1, 10)                                                                                  | (1,-1)                                               | (1,10)                       |
| result             | ![[report-3.png]]                                                                        | ![[report-4.png]]                                    | ![[report-5.png]]            |
| remark             | Lost track when occluded but being able to retrack after reapperance, and almost no lag. | Using model 1 will highly demand the intial velocity | Decrease in `sigma_position` will lead to lost of track after the object is occluded|  


Questions:
• What is the effect of using a constant velocity motion model?
Accurate estimate of the initial velocity will increase the accuracy. However, uncorrect initial velocity will lead to error that could not be amende thereafter. 
And this model will lead to great lag after the object is occluded.

• What is the effect of assuming decreased/increased system noise?
Decreasing the value of the noise increased the performance of the model.

• What is the effect of assuming decreased/increased measurement noise?
Increased measurement noise will lead to lost of the object, and tracking of nothing


### Video 3


|                    | Experiment 1                       | Experiment 2                                              | Experiment 3      |
| ------------------ | ---------------------------------- | --------------------------------------------------------- | ----------------- |
| `hist_bin`         | 16                                 | 16                                                        | 16                |
| `alpha`            | 0                                  | 0                                                         | 0.8               |
| `sigma_observe`    | 0.1                                | 0.1                                                       | 0.1               |
| `model`            | 0                                  | 1                                                         | 1                 |
| `num_particles`    | 300                                | 300                                                       | 300               |
| `sigma_position`   | 15                                 | 15                                                        | 10                |
| `sigma_velocity`   | 1                                  | 5                                                         | 1                 |
| `initial_velocity` | (1, 10)                            | (1,10)                                                    | (10,0)            |
| result             | ![[report-6.png]]                  | ![[report-7.png]]                                         | ![[report-8.png]] |
| remark             | Still being able to track the ball | Increase the velocity variance, the performance is better | Using model 1 and the correct intial velocity estimation                  |

Using the same parameters as video2, the model is still able to track the ball. Since the ball is never occluded. But the performance is not as good since the ball has fairly large variance in velocity.

Questions:
• What is the effect of using a constant velocity motion model?
The conclusion is not affected. Besides, using model 1 will have larger variance even if we have the correct initial velocity estimation.

• What is the effect of assuming decreased/increased system noise?
The conclusion is not affected. 

• What is the effect of assuming decreased/increased measurement noise?
The conclusion is not affected. 

## Questions
1. What is the effect of using more or fewer particles?
Usually more particles will lead to better result but more computational cost. 

2. What is the effect of using more or fewer bins in the histogram color model?
Same as the above question. But after a certain number, more bins will not increase the performance.

3. What is the advantage/disadvantage of allowing appearance model updating?
If the object itself is changing it will be helpful to allow bigger `alpha`. However if the object is occluded during the motion, we need to be careful since it may lead to the change of object being tracked.