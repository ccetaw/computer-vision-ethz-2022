# Computer Vision Exercise 5 Report
*Name*: Jingyu Wang
*Legi*: 21-956-453

## Image Segmentation
### `distance()` 
**for-loop-based:**  
Take the difference of `X` and `x`, `torch` will take care of the dimension, and we could simply take the norm. 

**Vectorized version:**
We aim to calculate the distance matrix. I used the idea in lab02, i.e. the function below

```python
def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    p2 = np.sum(desc1 * desc1, axis = 1, keepdims=True) * np.ones((desc1.shape[0], desc2.shape[0]))
    q2 = np.sum(desc2 * desc2, axis = 1, keepdims=True).transpose() * np.ones((desc1.shape[0], desc2.shape[0]))
    pq = desc1 @ desc2.transpose()
    distances = p2 - 2*pq + q2
    return distances
```
Replace `desc1` and `desc2` with `X` will work. 

### `gaussian()`
Simple implementation following 

$$f(x|\mu,\sigma^2)=\frac{1}{(2\pi)^{1/2}\sigma}\exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right]$$

with $\mu = 0$ and $\sigma=$ `bandwidth`. 
In both for-loop-based and vectorized version, the function implementation is the same, but results have dimension #pixels x 1 and #pixels x #pixels respectively.

*The normalization term won't afftect the result since in the `update_point()` step the weights will be divided by `weights.sum()` and the normalization term will be eliminated.* 

### `update_point()`
Simply multiply the weight with `X` using matrix multiplication. 

### `meanshift_step()`
In vectorized version, no loop is needed. Since `weight.matmul(X)` will directly give the result. 


