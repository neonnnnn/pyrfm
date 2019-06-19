# pyrfm
A library for random feature maps in Python.

Random feature maps are promising methods for large scale kernel methods.
They are maps from a original feature space to a randomized feature space 
approximating a kernel induced feature space.
The idea is to run linear models on such randomized feature space.
When the dimension of the random feature map D is not so high and the number of
training example N is large, this approach is very efficient compared to 
canonical kernel methods.

pyrfm follows the scikit-learn API and supports following random features:

 - random Fourier feature (for the RBF kernel) [1]
 - signed circulant random Fourier feature (for the RBF kernel) [2]
 - random Maclaurin feature (for the polynomial kernel, exp kernel, and 
 user-specified dot product kernels) [3]
 - tensor sketching (for the polynomial kernel) [4]
 - random kernel feature (for the ANOVA kernerl and all-subsets kernel) [5]
 - signed circulant random kernel feature (for the ANOVA kernel) [5]
 - S.Maji and A.Berg feature (for the intersection (min) kernel) (this feature 
 is not random) [6]
  
The random Fourier feature is also implemented in scikit-learn 
(kernel_approximation.RBFSampler).

Moreover, pyrfm supports following solvers for linear models:
 - primal coordinate descent for sparse S.Maji and A.Berg feature [6,7]
 - AdaGrad for very-large-scale dataset: does not compute the random feature map
  of all examples at the same time (slow but space efficient) [8]
 
 # Installation
 1. Download the source codes by
 
    git clone https://github.com/neonnnnn/pyrfm.git
 
   or download as a ZIP from GitHub.
 
 2. Install the dependencies:
 
    cd pyrfm
    
    pip install -r requirements.txt
    
 3. Finally, build and install pyrfm by
 
    python setup.py install
    
 # References
    [1] Ali Rahmini and Ben Recht. Random Feature Maps for Large-Scale Kernel 
    Machines. In Proc. NIPS, pp. 1177-1184, 2007.
    
 # Authors
 - Kyohei Atarashi, 2018-present