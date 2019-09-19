from .random_fourier import RandomFourier
from .random_kernel import RandomKernel, RandomSubsetKernel
from .random_maclaurin import RandomMaclaurin
from .signed_circulant_random_kernel import SignedCirculantRandomKernel
from .maji_berg import MB, SparseMB
from .tensor_sketch import TensorSketch
from .compact_random_feature import CompactRandomFeature
# for structured random features
from .fastfood import FastFood
from .subsampled_random_hadamard import SubsampledRandomHadamard
from .signed_circulant_random_projection import SignedCirculantRandomMatrix
from .random_projection import RandomProjection


DOWNSAMPLES = {'srht': SubsampledRandomHadamard,
               None: RandomProjection}
