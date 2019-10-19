# pyrfm
A library for random feature maps and linear models with random feature maps in Python.

## Installation
 1. Download the source codes by
 
 
    git clone https://github.com/neonnnnn/pyrfm.git
 
  or download as a ZIP from GitHub.
  
 2. Install the dependencies:
 
 
    cd pyrfm
    
    pip install -r requirements.txt
    
 3. Finally, build and install pyrfm by
 
 
    python setup.py install


## What are random feature maps?
Random feature maps are promising methods for large-scale kernel methods.
They are maps from a original feature space to a randomized feature space 
approximating a kernel-induced feature space.
The idea is to run linear models on such randomized feature space for 
classification, regression, clustering, etc.
When the dimension of the random feature map D is not so high and the number of
training example N is large, this approach is very efficient compared to 
canonical kernel methods.

## Random Feature Maps Implemented
pyrfm follows the scikit-learn API and now **supports following random features**.

 - `RandomFourier`: random Fourier feature (for the RBF kernel) [1]
 - `RandomMaclaurin`: random Maclaurin feature (for the polynomial kernel, exp kernel, and 
 user-specified dot product kernels) [3]
 - `TensorSketch`: tensor sketching (for the polynomial kernel) [4]
 - `RandomKernel`: random kernel feature (for the ANOVA kernel and all-subsets kernel) [5]
 - `MB`: S.Maji and A.Berg feature (for the intersection (min) kernel) (this feature 
 is not random) [6]
 
In other words, pyrfm now **provides approximaters for following kernels**.
 - RBF kernel (random Fourier)
 - polynomial kernel (random Maclaurin, tensor sketching)
 - exponential kernel (random Maclaurin)
 - user-specified dot product kernel (random Maclaurin, requiring Maclaurin 
 coefficients of specified kernel)
 - ANOVA kernel (random kernel, signed circulant random kernel)
 - all-subsets kernel (random kernel)
 - intersection (min) kernel (S.Maji and A.Berg)
  
The random Fourier feature is also implemented in scikit-learn 
(`kernel_approximation.RBFSampler`).

Furthermore, pyrfm **supports following structured random features**.
 - `SignedCirculantRandomMatrix`: signed circulant random matrix (for the dot product / RBF kernel) [2]
 - `SignedCirculantRandomKernel`: signed circulant random kernel feature (for the ANOVA kernel) [5]
 - `SubsampledRandomHadamard`: subsampled random Hadamard transform (for the dot product) [11]
 - `FastFood`: fastfood (for the dot product / RBF kernel) [12]
 - `CompactRandomFeature`: compact random features [13] (with subsampled ranadom Hadamard transform or random projection [14])
 - `OrthogonalRandomFeature` / `StructuredOrthogonalRandomFeature`: orthogonal random feature / structured orthogonal random feature (for the dot product / RBF kernel) [15]

These methods are faster and more memory-efficient than canonical random features such as random Fourier, random kernel, etc.
We believe that you can use these structured random features as a subroutine of your proposed random features.

## Linear Models Implemented
Moreover, pyrfm **supports following solvers for linear models with random features**.
 - `SparseMBRegressor` / `SparseMBClassifier`: primal coordinate descent for sparse S.Maji and A.Berg feature [6,7]
 - `AdaGradRegressor` / `AdaGradClassifier`: AdaGrad for very-large-scale dataset: does not compute the random feature map
  of all examples at the same time (space efficient but slow) [8]
 - `SDCARegressor` / `SDCAClassifier`: SDCA for very-large-scale dataset: does not compute the random feature map
  of all examples at the same time (space efficient but slow) [9]
 - `AdamRegressor` / `AdamClassifier`: Adam for very-large-scale dataset: does not compute the random feature map
  of all examples at the same time (space efficient but slow) [10]
 - `SGDRegressor` / `SGDClassifier`: SGD/ASGD for very-large-scale dataset: does not compute the random feature map
  of all examples at the same time (space efficient but slow) [16, 17]
 - `SAGARegressor` / `SAGAClassifier`: SAG/SAGA for very-large-scale dataset: does not compute the random feature map
  of all examples at the same time (space efficient but slow) [18, 19]
  
 All methods support squared loss for regression and hinge loss, squared hinge loss, and logistic loss for classification.
 
 AdaGrad, SDCA, Adam, SGD/ASGD, and SAG/SAGA in pyrfm are for a very-large-scale dataset such that computing its random feature matrix (i.e., computing random features for all instances at the same time)
 causes MemoryError.
 If you can allocate memory for random feature matrix of your training data, you should use the other implementations of linear models (linear_model in scikit-learn, sklearn-contrib-lightning, etc). 
 Now, these stochastic solvers run efficiently for following random features.
  - `RBFSampler` (in sklearn.kernel_approximation)
  - `RandomFourier`
  - `RandomMaclaurin`
  - `TensorSketch`
  - `RandomKernel`
  - `SignedCirculantRandomProjection`
  - `FastFood`
  - `SubsampledRandomHadamardTransform`
  - `CompactRandomFeature`
  - `OrthogonalRandomFeature` / `StructuredOrthogonalRandomFeature`
  
 
 For improving efficiency, implement cdef class and cdef transform method for your desired transformer.
 Please see random_feature/random_features_fast.pyx/pxd.
 Although these stochastic solvers **support any transformers, they might run unbelievable slow** when there is no cdef class and cdef transform method for your desired transformer in random_features_fast.pyx/pxd.
 We believe that these implementations can be used for researches.
     
 ## References
    [1] Ali Rahimi and Ben Recht. Random Feature Maps for Large-Scale Kernel Machines. 
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
        JMLR, vol. 12, pp. 2121--2159, 2012.
    [9] Shai Shalev-Shwartz and Tong Zhang. Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization.
        JMLR, vol. 14, pp. 567--599, 2013.
    [10] Diederik P. Kingma and Jimmy Lei Ba. Adam: A Method for Stochastic Optimization.
         In Proc. ICLR, 2015.
    [11] Joel A. Tropp. Improved Analysis of the Subsampled Randomized Hadamard Transform.
         Adv. Adapt. Data AnAl., vol. 3, num. 1-2, pp. 115--126, 2011.
    [12] Quoc Le, Tamas Sarlos, and Alex Smola. Fastfood — Approximating Kernel Expansions in Loglinear Time.
         In Proc. ICML, pp. 244--252, 2013.
    [13] Raffay Hamid, Ying Xiao, Alex Gittens, and Dennis DeCoste. Compact Random Feature Maps.
         In Proc. ICML, pp. 19--27, 2014.
    [14] Ping Li, Trevor J. Hastie, and Kenneth W. Church. Very Sparse Random Projections.
         In Proc. KDD, pp. 287--296, 2006.
    [15] Felix Xinnan Yu, Ananda Theertha Suresh, Krzysztof Choromanski, Daniel Holtmann-Rice, and Sanjiv Kumar. Orthogonal Random Features.
         In Proc. NIPS, pp. 1975--1983, 2016.
    [16] Leon Bottou. Large-Scale Machine Learning with Stochastic Gradient Descent.
         In Proc. COMPSTAT', pp. 177--186, 2010.
    [17] Leon Bottou. Stochastic Gradient Descent Tricks.
         Neural Networks, Tricks of the Trade, Reloaded. pp. 430--445, 
         Lecture Notes in Computer Science (LNCS 7700), Springer, 2012
    [18] Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives.
         In Proc. NIPS 2014, pp. 1646--1654, 2014.
    [19] Mark Schmidt, Nicolas Le Roux, and Francis Bach. Minimizing Finite Sums with the Stochastic Average Gradient.
         Mathematical Programming Vol 162, pp. 83--112, 2017.
  
 # Authors
 - Kyohei Atarashi, 2018-present