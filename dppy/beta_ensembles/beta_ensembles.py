from abc import ABCMeta, abstractmethod
from re import findall


class AbstractBetaEnsemble(metaclass=ABCMeta):
    """:math:`\\beta`-Ensemble object parametrized by

    :param beta:
        :math:`\\beta >= 0` inverse temperature parameter.

        The default :py:attr:`beta`:math:`=2` corresponds to the DPP case,
        see :ref:`beta_ensembles_definition_OPE`
    :type beta:
        int, float, default :math:`2`

    .. seealso::

        - :math:`\\beta`-Ensembles :ref:`definition <beta_ensembles_definition>`
    """

    def __init__(self, beta=2):

        if not (beta >= 0):
            raise ValueError("`beta` must be >=0. Given: {}".format(self.beta))
        self.beta = beta

        # Split object name at uppercase
        self.name = " ".join(findall("[A-Z][^A-Z]*", self.__class__.__name__))
        self.params = {"size_N": 10}  # Number of points and ref measure params

        self.sampling_mode = ""
        self.list_of_samples = []

    @property
    def _str_title(self):
        return r"Realization of {} points of {} with $\beta={}$".format(
            self.params["size_N"], self.name, self.beta
        )

    def __str__(self):
        str_info = (
            "{} with beta = {}".format(self.name, self.beta),
            "sampling parameters = {}".format(self.params),
            "number of samples = {}".format(len(self.list_of_samples)),
        )

        return "\n".join(str_info)

    def flush_samples(self):
        """Empty the :py:attr:`list_of_samples` attribute."""
        self.list_of_samples = []

    @abstractmethod
    def sample_full_model(self):
        """Sample from underlying :math:`\\beta`-Ensemble using the corresponding full matrix model.
        Arguments are the associated matrix dimensions
        """

    @abstractmethod
    def sample_banded_model(self):
        """Sample from underlying :math:`\\beta`-Ensemble using the corresponding banded matrix model.
        Arguments are the associated reference measure's parameters, or the matrix dimensions used in :py:meth:`sample_full_model`
        """

    @abstractmethod
    def plot(self):
        """Display last realization of the underlying :math:`\\beta`-Ensemble.
        For some :math:`\\beta`-Ensembles, a normalization argument is available to display the limiting (or equilibrium) distribution and scale the points accordingly.
        """

    @abstractmethod
    def hist(self):
        """Display histogram of the last realization of the underlying :math:`\\beta`-Ensemble.
        For some :math:`\\beta`-Ensembles, a normalization argument is available to display the limiting (or equilibrium) distribution and scale the points accordingly.
        """

    @abstractmethod
    def normalize_points(self):
        """Normalize points ormalization argument is available to display the limiting (or equilibrium) distribution and scale the points accordingly."""
