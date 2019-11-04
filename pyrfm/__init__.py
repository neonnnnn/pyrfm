from .random_feature import (RandomFourier, RandomMaclaurin,
                             SubfeatureRandomMaclaurin,
                             RandomKernel, SubfeatureRandomKernel,
                             SignedCirculantRandomKernel,
                             TensorSketch, MB, SparseMB, RandomProjection,
                             FastFood, SubsampledRandomHadamard,
                             SignedCirculantRandomMatrix,
                             CompactRandomFeature,
                             OrthogonalRandomFeature,
                             StructuredOrthogonalRandomFeature,
                             AdditiveChi2Sampler,
                             LearningKernelwithRandomFeature
                             )
from .kernels import (anova, all_subsets, chi_square, hellinger,
                      intersection, anova_fast, kernel_alignment)
from .kernels_fast import (_anova, _all_subsets, _chi_square,
                           _intersection, score)
from .linear_model import (AdaGradRegressor, AdaGradClassifier,
                           SparseMBRegressor, SparseMBClassifier,
                           SDCARegressor, SDCAClassifier,
                           AdamRegressor, AdamClassifier,
                           SGDRegressor, SGDClassifier,
                           SAGARegressor, SAGAClassifier)


__all__ = ['RandomKernel', 'SubfeatureRandomKernel', 'RandomFourier',
           'TensorSketch', 'RandomMaclaurin', 'MB', 'SparseMB',
           'SubfeatureRandomMaclaurin', 'RandomProjection', 'FastFood',
           'CompactRandomFeature', 'SubsampledRandomHadamard',
           'SignedCirculantRandomMatrix', 'SignedCirculantRandomKernel',
           'OrthogonalRandomFeature', 'StructuredOrthogonalRandomFeature',
           'AdditiveChi2Sampler', 'LearningKernelwithRandomFeature',
           'AdaGradRegressor', 'AdaGradClassifier', 'AdamRegressor',
           'AdamClassifier', 'SparseMBRegressor', 'SparseMBClassifier',
           'SDCARegressor', 'SDCAClassifier', 'SGDClassifier', 'SGDRegressor',
           'SAGARegressor', 'SAGAClassifier']
