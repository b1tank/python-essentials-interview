# Numpy, Scipy, Matplotlib

## Array creation
```py
a = np.array([1,2,3])
a = np.arange(1,10,2)
a = np.linspace(0,1,6)  # [0, 0.2, 0.4, 0.6, 0.8, 1]
a = np.linspace(0,1,6,endpoint = False)  # [0, 0.2, 0.4, 0.6, 0.8]
a = np.ones([3,3]) # np.zeros
a = np.eye(3)
a = np.diag(np.array([1,2,3]))
a = np.random.rand(4,3) # uniform in [0,1]
a = np.random.randn(4,3) # Gaussian
a = np.random.randint(0, 21, 15)    # 15 ints drawn from [0, 21)
a = np.random.standard_normal(b.shape)
a = np.random.choice(pool,K)        # Random choose k elements from the pool
np.tile(a,[1,1,2])
np.concatenate((a1,a2,a3),axis = 0) # Specify axis. Default=0. Axis=None to flatten first before concatenation
x,y = np.mgrid[0:5, 0:5]        # dense meshgrid: x.shape = (5,5), y.shape = (5,5)
x,y = np.ogrid[0:5, 0:5]        # open meshgrid: x.shape = (5,1), y.shape = (1,5)
np.hstack((a,b))                # Return a copy, not a view
np.vstack((a,b))
```

### Reshaping
* All operations here always create a new VIEW (unless noted)!!! Original array shape not changed (unless noted).
```py
a.ravel()           # flatten the array, concatenate all rows together. Returns a view
a.flatten()         # Returns a copy!!!
a.reshape((w,h))
a.reshape((w,-1))   # automatic deduce h 
a[:,np.newaxis]     # Add a new dimension
a.T                 # transpose (not for 1D array)
a = np.zeros([4,3,2])
b = a.transpose(1,2,0)
b.shape             # --> (3,2,4)
a.resize((8,))      # Works inplace!!!
np.resize(a,(8,))   # Returns a new array (no memory sharing)!!!
```
* Resizing doesn't handle multi-dimensional image well.


## Check array information & manipulating array type
```py
a = np.ones([3,4])
a.dim   # 2
a.shape # (3,4)
a.size  # 12
len(a)  # 3   -> size of first dimension
a.dtype # default type is float
a = np.array([1,2,3], dtype = float)
b = a.astype(int)
# Cautious: using `a.dtype = int` doesn't do the job! It's like reinterpret_cast
np.array([1,2,3])+0.0  # operation may change array type, ...
b[0] = 1.9             # but assignment doesn't. (b[0] will be 1 if a is int type)
c = np.around(a).astype(int)
```

## Slicing rule
* Slicing operation shares the view (unlike built-in list)! Use .copy() to force a copy.
    * Can use np.may_share_memory(a,b) to check
* Python-stype slicing still works
* Use boolean masks
```py
a = np.random.randint(0, 21, 15)
extract_from_a = a[a % 3 == 0]
```
* Indexing with an array of integers
    * 2D array: a[0][1] is same as a[0,1]
    * 2D built-in list: a[0][1] is valid, but not a[0,1]
```py
a = np.array([0,2,4,6,8,10,12,14,16,18])
a[[2, 3, 2, 4, 2]]          # [4,6,4,8,4]
idx = np.array([[2, 3, 2],[4, 5, 6]])
a[idx]                      # [[4,6,4],[8,10,12]]
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a[(0,1),(0,1)]              # same as a[[0,1],[0,1]]  -> [1,5]
a[:2,(0,1)]                 # [[1,2],[4,5]]
a[np.array([1,0,1],dtype=bool),1]     # [2,8]
```

## Numpy array operations
* Array & scalar arithmatics: 
    * A numpy array can directly operate (using +-*/) with a scalar (through broadcasting)
    * A built-in array can't do *-/ with scalar. * means repitition 
* Array & array arithmatics:
    * Numpy arrays can do +-*/
    * Built-in arrays can't do -*/, + is concatenation
    * Direct */ is element-wise multiply/division
    * `np.multiply(a,b)` and `np.divide(a,b)` are all element-wise operation (may involve broadcasting)
    * Matrix multplication is `a.dot(b)` and `np.matmul(a,b)`
* Comparison (==, >, <)
    * Numpy array do element-wise comparison (return an array of booleans)
    * Built-in array do dictionary-order comparison (return a single boolean)
    * `np.array_equal(a,b) `check equality of two arrays
* Other element-wise operations
    * `np.logical_or(a,b)`, `np.logical_and(a,b)`
    * `np.sin(a)`, `np.log(a)`, `np.exp(a)`
    * `np.real(a), np.imag(a)`
    * `np.sign(a)`
* Matrix operations
    * `a.T   # transpose. Warning! transposition is a view!`
        * transpose a vector: `a = a[:, np.newaxis]`
    * `a.dot(b)`
    * Vector multiplication is `np.vdot(a,b)` (will reshape a,b to vector if they are not)
    * `np.inner(a,b), np.outer(a,b)`
* Reduction operations
    * `np.sum(a)`, `a.sum()`       # sum of all valus even if high-dimensional
    * `a.sum(axis=0)`, `np.sum(a, axis=0)`     # sum columns if axis=0, sum rows if axis=1. The specified axis is reduced to a scalar
    * `np.min(a)`, `np.max(a)`, `a.min()`, `a.max()`
    * `np.argmin(a)`, `np.argmax(a)`, `a.argmin()`, `a.argmax()`   # a single index, even if for high-dimensional arrays (flatten it first). Find the first index
    * `np.all(), np.any(), a.all(), a.any()`
    * `np.mean(a), np.median(a), np.std(a), a.mean(), a.median(), a.std()`
    * All can specify axis

## Broadcasting
* Numpy uses broadcasting for shape mismatch. Different behaviors of list and array
    * x = [1,2,3,4,5]
    * x[1:2] = [10,10,10,10]  --> [1, 10, 10, 10, 10, 3, 4, 5]
    * x = np.array([1,2,3,4,5])
    * x[1:2] = [10,10,10,10]  --> ValueError: could not broadcast input array from shape (4) into shape (1)
    * `x,y = np.ogrid[0:5, 0:5]`    `x, y = np.mgrid[0:5,0:5]`


## Sorting
```py
np.sort(a,axis=1)   # default axis=-1. Return a new copy
a.sort()            # works in place
j = np.argsort(a)   # be the indexes for sorting
a[j]                # sorted array
```



## Linear algebra
* Polynomial fitting using numpy
```py
x = np.linspace(0, 1, 20)
y = np.cos(x) + 0.3*np.random.rand(20)
p = np.poly1d(np.polyfit(x, y, 3))      # np.poly1d([3,1,2]) means 3x^2+x+2
```
* Scipy.linalg
 ```py
 from scipy import linalg
 detval = linalg.det(a)
 uarr, spec, vharr = linalg.svd(arr)    # spec is 1D array: 
                                        # uarr.dot(np.diag(spec)).dot(vharr) = arr
 ainv = linalg.inv(a)
 ainv = linalg.pinv(a)     # least square solver, pinv2 uses svd
 x = linalg.norm(a)     # Frobenius norm
 x = linalg.solve(A,b)  # A has to be square
 x = linalg.lstsq(A,b)
 la, v = linalg.eig(A)
 la, v = linalg.eigvals(A)
```



## IO
* Numpy Arrays
```py
np.savetxt('fn.txt',data)
data2 = np.loadtxt('fn.txt')
np.save('fn2.npy',data)
data3 = np.load('fn2.npy')
```
* Images using matplotlib
```py
import matplotlib.pyplot as plt
img = plt.imread('img.png')
plt.imshow(img)
plt.savefig('newimg.png')
plt.imsave('red.png',img[:,:,0],cmap=plt.cm.gray)
```
* Scipy IO
* Warning: mat file doesn't save 1D arrays. It will be automatically converted to 2D: [1,2,3] --> [[1,2,3]]
```py
from scipy import io as spio
spio.savemat('file.mat', {'a': a})
data = spio.loadmat('file.mat')
data['a']
from scipy import misc
img = misc.imread('fname.png')
```



## Scipy libraries
* Interpolate
```py
from scipy.interpolate import interp1d
linear_interp = interp1d(x1, y1)
x2 = np.linspace(0, 1, 50)             # Finer linspace than x1
y2 = linear_interp(x2)
cubic_interp = interp1d(x1, y1, kind='cubic')
y3 = cubic_interp(x2)
```
* Optimization: fit function, minimize, root of equation
```py
from scipy import optimize
def test_func(x, a, b):
    return a * np.sin(b * x)
params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
                    # params get estimates for a & b
                    
def f(x):
    return x**2 + 10*np.sin(x)
result = optimize.minimize(f, x0=0, method="L-BFGS-B", bounds=((0, 10), )) # method & bounds are optional
result.x            # array([-1.30644...])
result2 = optimize.basinhopping(f, 0) # global minimum? combines a local optimizer with sampling of starting points

root = optimize.root(f, x0=1)
root.x              # array([ 0.])
```
* Statistics: probability density function & probability fitting, percentile
```py
from scipy import stats
bins = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
pdf = stats.norm.pdf(bins)
plt.plot(bins, pdf)
samples = np.random.normal(size=1000)
loc, std = stats.norm.fit(samples)
stats.scoreatpercentile(samples, 90)    # 90% percentile
stats.ttest_ind(a, b)                   # T statistic value & p value (the probability of both processes being identical)
```
* Integration & ODE solver
```py
from scipy.integrate import quad
res, err = quad(np.sin, 0, np.pi/2)     # Integrate from 0 to pi/2 for sin(x). res is result, err is estimated error. 

def calc_derivative(ypos, time):
    return -2 * ypos
from scipy.integrate import odeint
time_vec = np.linspace(0, 4, 40)
y = odeint(calc_derivative, y0=1, t=time_vec)
```
* FFT
```py
from scipy import fftpack
sig_fft = fftpack.fft(sig)
freqs = fftpack.fftfreq(sig.size, d=time_step)
sig_recon = fftpack.ifft(sig_fft)
```

## Image manipulation using scipy.ndimage
* Image translation, rotation, cropping and zooming
```py
from scipy import ndimage
shifted_face = ndimage.shift(face, (50, 50))
shifted_face2 = ndimage.shift(face, (50, 50), mode='nearest')
rotated_face = ndimage.rotate(face, 30)
cropped_face = face[50:-50, 50:-50]
zoomed_face = ndimage.zoom(face, 2)
```
* Image filtering
```py
blurred_face = ndimage.gaussian_filter(noisy_face, sigma=3)
median_face = ndimage.median_filter(noisy_face, size=5)
wiener_face = scipy.signal.wiener(noisy_face, (5, 5))
```
* Mathematical morphology
    * For gray-valued images, eroding (resp. dilating) amounts to replacing a pixel by the minimal (resp. maximal) value
among pixels covered by the structuring element centered on the pixel of interest. (Api will be gray_erosion, etc)
```py
# structuring element
el = ndimage.generate_binary_structure(2, 1)    # Two arguments are rank and connectivity. For detailed explanation: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.generate_binary_structure.html
el = np.ones((3,3))
# input object
a = np.zeros((7, 7), dtype=np.int)
a[1:6, 2:5] = 1
# Morphology operations
erosion1 = ndimage.binary_erosion(a).astype(a.dtype)    # If no structuring element is provided, an element is generated with a square connectivity equal to one. Default dtype is bool.
erosion2 = ndimage.binary_erosion(a, structure = el).astype(a.dtype)
dilation = ndimage.binary_dilation(a).astype(a.dtype)
ndimage.binary_opening(a, structure=np.ones((3, 3))).astype(np.int)
ndimage.binary_closing(a, structure=np.ones((3, 3))).astype(np.int)
```
* Connected components
```py
labels, nb = ndimage.label(mask)    # mask is a binary mask containing several connected components
                                    # nb is the number of labels
areas = ndimage.sum(mask, labels, range(1, labels.max()+1))     # sum in each connected component
maxima = ndimage.maximum(sig, labels, range(1, labels.max()+1)) # maximum in each connectd component
sl = ndimage.find_objects(labels==4)
plt.imshow(sig[sl[0]])
```

## Data visualization
```py
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6),dpi=80)  # do not always have to do it
plt.subplot(1,2,1)
plt.plot(x,y)   #plt.plot(x,y,'-o')
plt.plot(x,y,color="blue", linewidth=1.0, linestype = "-", label="SomeCurve")
plt.xlim(-1.0,1.0)
plt.xlim(0.0,10.0)
plt.xticks(np.linspace(-1.0,1.0,9))
plt.legend(loc='upper left')
plt.subplot(1,2,2)
plt.show()
plt.imshow(np.random.rand(100,100))
plt.imshow(Z, interpolation='nearest', cmap='bone', origin='lower')
plt.axis('off')
plt.colorbar()
plt.show()
```

## Scikit-image (Fill in later...)
