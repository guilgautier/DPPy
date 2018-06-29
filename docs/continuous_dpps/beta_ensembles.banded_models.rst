.. _banded_matrix_models:

Banded models
~~~~~~~~~~~~~

Computing the eigenvalues of a full :math:`N\times N` random matrix is :math:`\mathcal{O}(N^3)` can become prohibitive for large :math:`N`.
A way to circumvent the problem is to adopt the equivalent banded models i.e. diagonalize banded matrices.

The first tridiagonal models for the :ref:`hermite_ensemble: and :ref:`laguerre_ensemble: were revealed by :cite:`DuEd02`, who left the :ref:`jacobi_ensemble: one as an open question, addressed by :cite:`KiNe04`.
These tridiagonal formulation permit sampling in :math:`\mathcal{O}(N^2)` but also unlocked generic :math:`\beta>0`!

:cite:`KiNe04` also derived a quindiagonal model for the :ref:`circular_ensemble`

.. seealso::

	- Hermite, :cite:`DuEd02` II-C

	- Laguerre, :cite:`DuEd02` III-B

	- Jacobi, :cite:`KiNe04` Theorem 2

	- Circular, :cite:`KiNe04` Theorem 1