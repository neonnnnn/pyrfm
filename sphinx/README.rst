pyrfm
=====

A library for random feature maps and linear models with random feature
maps in Python.

Installation
------------

1. Download the source codes by::

    git clone https://github.com/neonnnnn/pyrfm.git

or download as a ZIP from GitHub.

2. Install the dependencies:::

    cd pyrfm

    pip install -r requirements.txt

3. Finally, build and install pyrfm by::

    python setup.py install


For running example codes (in pyrfm/benchmarks), jupyter and matplotlib are required.

What are random feature maps?
-----------------------------

Using random feature maps is a promising way for large-scale kernel methods.
They are maps from an original feature space to a randomized feature space approximating a kernel-induced feature space.
The basic idea is to run linear models on such randomized feature space for classification, regression, clustering, etc.
When the dimension of the random feature map D is not so high and the number of training example N is large, this approach is very efficient compared to canonical kernel methods.

Random Feature Maps Implemented
-------------------------------

.. currentmodule:: pyrfm.random_feature

pyrfm follows the scikit-learn API and now **supports following random features**.

- :class:`RandomFourier`: random Fourier feature (for the RBF kernel) :cite:`rahimi2008random`

- :class:`RandomMaclaurin`: random Maclaurin feature (for the polynomial kernel, exp kernel, and user-specified dot product kernels) :cite:`kar2012random`

- :class:`TensorSketch`: tensor sketching (for the polynomial kernel) :cite:`pham2013fast`

- :class:`RandomKernel`: random kernel feature (for the ANOVA kernel and all-subsets kernel) :cite:`atarashi2019random`

- :class:`MB`: S.Maji and A.Berg feature (for the intersection (min) kernel) (this feature is not random) :cite:`maji2009max`

In other words, pyrfm now **provides approximaters for following kernels**.

- RBF kernel (:class:`RandomFourier`)

- polynomial kernel (:class:`RandomMaclaurin`, :class:`TensorSketching`)

- exponential kernel (:class:`RandomMaclaurin`)

- user-specified dot product kernel (:class:`RandomMaclaurin`, requiring Maclaurin coefficients of specified kernel)

- ANOVA kernel (:class:`RandomKernel`)

- all-subsets kernel (:class:`RandomKernel`)

- intersection (min) kernel (:class:`MB`)

The random Fourier feature is also implemented in scikit-learn (``kernel_approximation.RBFSampler``).

Furthermore, pyrfm **supports following structured random features**.

- :class:`SignedCirculantRandomMatrix`: signed circulant random matrix (for the dot product / RBF kernel) :cite:`feng2015random`

- :class:`SignedCirculantRandomKernel`: signed circulant random kernel feature (for the ANOVA kernel) :cite:`atarashi2019random`

- :class:`SubsampledRandomHadamard`: subsampled random Hadamard transform (for the dot product) :cite:`tropp2011improved`

- :class:`FastFood`: fastfood (for the dot product / RBF kernel) :cite:`le2013fastfood`

- :class:`CompactRandomFeature`: compact random features :cite:`hamid2014compact` (with subsampled ranadom Hadamard transform or random projection :cite:`li2006very`)

- :class:`OrthogonalRandomFeature` / :class:`StructuredOrthogonalRandomFeature`: orthogonal random feature / structured orthogonal random feature (for the dot product / RBF kernel) :cite:`yu2016orthogonal`

These methods are faster and more memory-efficient than canonical random features such as random Fourier, random kernel, etc.
We believe that you can use these structured random features as a subroutine of your proposed random features, and :class:`SignedCirculantRandomKernel` is an example of it (:class:`SignedCirculantRandomMatrix` is used as a subroutine).

Moreover, following data-dependent random feature methods are supported:

- :class:`LearningKernelwithRandomFeature` :cite:`sinha2016learning`

Additionaly, we provide following random projection methods.

- :class:`RandomProjection` :cite:`li2006very`

- :class:`CountSketch` :cite:`weinberger2009feature` :cite:`charikar2002finding`

Linear Models Implemented
-------------------------

.. currentmodule:: pyrfm.linear_model

Moreover, pyrfm **supports following solvers for linear models with random features**.

- :class:`SparseMBRegressor` / :class:`SparseMBClassifier`: primal coordinate descent for sparse S.Maji and A.Berg feature :cite:`maji2009max` :cite:`chang2008coordinate`

- :class:`AdaGradRegressor` / :class:`AdaGradClassifier`: AdaGrad for very-large-scale dataset: does not compute the random feature map of all examples at the same time (space efficient but slow) :cite:`duchi2011adaptive`

- :class:`AdamRegressor` / :class:`AdamClassifier`: Adam for very-large-scale dataset: does not compute the random feature map of all examples at the same time (space efficient but slow) :cite:`kingma2014adam`

- :class:`DoublySGDRegressor` / :class:`DoublySGDClassifier`: Doubly stochastic gradient method proposed in :cite:`dai2014scalable`.

- :class:`SDCARegressor` / :class:`SDCAClassifier`: SDCA for very-large-scale dataset: does not compute the random feature map of all examples at the same time (space efficient but slow) :cite:`shalev2013stochastic`

- :class:`SGDRegressor` / :class:`SGDClassifier`: SGD/ASGD for very-large-scale dataset: does not compute the random feature map of all examples at the same time (space efficient but slow) :cite:`bottou2010large` :cite:`bottou2012stochastic`

- :class:`SAGARegressor` / :class:`SAGAClassifier`: SAG/SAGA for very-large-scale dataset: does not compute the random feature map of all examples at the same time (space efficient but slow) :cite:`defazio2014saga` :cite:`schmidt2017minimizing`

All methods support squared loss for regression and hinge loss, squared hinge loss, and logistic loss for classification.

AdaGrad, SDCA, Adam, SGD/ASGD, and SAG/SAGA in pyrfm are for a very-large-scale dataset such that computing its random feature matrix (i.e., computing random features for all instances at the same time) causes MemoryError.
If you can allocate memory for random feature matrix of your training data, you should use the other implementations of linear models (linear\_model in scikit-learn, sklearn-contrib-lightning, etc).
DoublySGD solvers do not keep the random weights but samples them at each iteration.
DoublySGD solver increase the number of random weights vector at every iteration.
We recommend batch_size > 1 and n_bases_sampled > 1.
For more details, please see :cite:`dai2014scalable`.


.. currentmodule:: pyrfm.random_feature

Now, these stochastic solvers except DoublySGD solvers run efficiently for following random features.

- ``RBFSampler`` and `SkewedChi2Sampler` (in sklearn.kernel\_approximation)

- :class:`RandomFourier`

- :class:`RandomMaclaurin`

- :class:`TensorSketch`

- :class:`RandomKernel`

- :class:`SignedCirculantRandomProjection`

- :class:`FastFood`

- :class:`SubsampledRandomHadamardTransform`

- :class:`CompactRandomFeature`

- :class:`OrthogonalRandomFeature` / :class:`StructuredOrthogonalRandomFeature`

- :class:`AdditiveChi2Sampler` (wrapper of ``AdditiveChi2Sampler`` in sklearn.kernel\_approximation)

For improving efficiency, implement cdef class and cdef transform method for your desired transformer.
Please see ``random_feature/random_features_fast.pyx/pxd``.
Although these stochastic solvers **support any transformers, they might run unbelievable slow** when there is no cdef class and cdef transform method for your desired transformer in ``random_features_fast.pyx``.
DoublySGD solvers supports only :class:`RBFSampler`, :class:`RandomFourier`, :class:`RandomMaclaurin`, :class:`RandomKernel`, and ``SkewedChi2Sampler``. 
We believe that these implementations can be used for researches.


References
----------
.. bibliography:: refs.bib

Authors
-------
- Kyohei Atarashi, 2018-present
