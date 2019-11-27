from sklearn import kernel_approximation
class AdditiveChi2Sampler(kernel_approximation.AdditiveChi2Sampler):
    """A wrapper of sklearn.kernel_approximation.AdditiveChi2Sampler.
    
    Parameters
    ----------
    sample_steps : int, optional
        Gives the number of (complex) sampling points.

    sample_interval : float, optional
        Sampling interval. Must be specified when sample_steps not in {1,2,3}.
    
    References
    ----------
    See `"Efficient additive kernels via explicit feature maps"
    <http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf>`_
    A. Vedaldi and A. Zisserman.
    Pattern Analysis and Machine Intelligence, 2011.

    """

    def __init__(self, sample_steps=2, sample_interval=None):
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval

    def fit(self, X, y=None):
        self.n_features = X.shape[1]
        self.n_components = X.shape[1] * (2*self.sample_steps - 1)
        return super(AdditiveChi2Sampler, self).fit(X, y)
