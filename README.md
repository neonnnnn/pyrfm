# pyrfm
A library for random feature maps in Python.

Random feature maps are promising methods for large-scale kernel methods.
They are maps from a original feature space to a randomized feature space 
approximating a kernel induced feature space.
The idea is to run linear models on such randomized feature space for 
classification, regression, clustering, etc.
When the dimension of the random feature map D is not so high and the number of
training example N is large, this approach is very efficient compared to 
canonical kernel methods.

pyrfm follows the scikit-learn API and now **supports following random features**:

 - random Fourier feature (for the RBF kernel) [1]
 - signed circulant random Fourier feature (for the RBF kernel) [2]
 - random Maclaurin feature (for the polynomial kernel, exp kernel, and 
 user-specified dot product kernels) [3]
 - tensor sketching (for the polynomial kernel) [4]
 - random kernel feature (for the ANOVA kernel and all-subsets kernel) [5]
 - signed circulant random kernel feature (for the ANOVA kernel) [5]
 - S.Maji and A.Berg feature (for the intersection (min) kernel) (this feature 
 is not random) [6]
 
In other words, pyrfm now **provides approximaters for following kernels**:
 - RBF kernel (random Fourier, signed circulant random Fourier)
 - polynomial kernel (random Maclaurin, tensor sketching)
 - exponential kernel (random Maclaurin)
 - user-specified dot product kernel (random Maclaurin, requiring Maclaurin 
 coefficients of specified kernel)
 - ANOVA kernel (random kernel, signed circulant random kernel)
 - all-subsets kernel (random kernel)
 - intersection (min) kernel (S.Maji and A.Berg)
  
The random Fourier feature is also implemented in scikit-learn 
(kernel_approximation.RBFSampler).

Moreover, pyrfm **supports following solvers for linear models**:
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
    [1] Ali Rahmini and Ben Recht. Random Feature Maps for Large-Scale Kernel Machines. 
        In Proc. NIPS, pp. 1177--1184, 2007.
    [2] Chang Feng, Qinghua Hu, and Shizhog Liao. Random Feature Mapping with Signed Circulant Matrix Projection. 
        In Proc IJCAI, pp. 3490--3497, 2015.
    [3] Purushottan Kar and Harish Karnick. Random Feature Maps for Dot Product Kernels. 
        In Proc. AISTATS, pp. 583--591, 2012.
    [4] Ninh Pham and Rasmus Pagh. Fast and Scalable Polynomial Kernels via Explicit Feature Maps. 
        In Proc. KDD, pp. 239--247, 2013.
    [5] Kyohei Atarashi, Subhransu Maji, and Satoshi Oyama. Random Feature Maps for the Itemset Kernel. 
        In Proc. AAAI, 2019.
    [6] Subhransu Maji and Alexander C. Berg. Max-Margin Additive Classifiers for Detection. 
        In. Proc. ICCV, pp. 40--47, 2009. 
    [7] Kai-Wei Chang, Cho-Jui Hsieh, and Chih-Jen Lin. Coordinate Descent Method for Large-scale L2-loss Linear Support Vector Machines.
        JMLR, vol. 9, pp 1369-1398, 2008.
    [8] John Duchi, Elad Hazan, and Yoram Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
        JMLR, vol 12, pp. 2121--2159, 2012.
 # Authors
 - Kyohei Atarashi, 2018-present