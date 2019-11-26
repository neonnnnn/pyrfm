from .random_fourier import RandomFourier
from .random_kernel import RandomKernel, SubfeatureRandomKernel
from .random_maclaurin import RandomMaclaurin, SubfeatureRandomMaclaurin
from .signed_circulant_random_kernel import SignedCirculantRandomKernel
from .maji_berg import MB, SparseMB
from .count_sketch import CountSketch
from .tensor_sketch import TensorSketch
from .compact_random_feature import CompactRandomFeature
# for structured random features
from .fastfood import FastFood
from .subsampled_random_hadamard import SubsampledRandomHadamard
from .signed_circulant_random_projection import SignedCirculantRandomMatrix
from .random_projection import RandomProjection
from .orthogonal_random_feature import (OrthogonalRandomFeature,
                                        StructuredOrthogonalRandomFeature)
from .additivechi2sampler import AdditiveChi2Sampler
from .learning_kernel_with_random_feature_fast import proj_l1ball, proj_l1ball_sort
from .learning_kernel_with_random_feature import LearningKernelwithRandomFeature


DOWNSAMPLES = {'srht': SubsampledRandomHadamard,
               None: RandomProjection}
