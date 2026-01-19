pcsaft.pcsaft\_den
==================

.. currentmodule:: pcsaft

.. autofunction:: pcsaft_den

Notes
-----
``pcsaft_den`` remains available as the explicit density solver (T, P -> rho).
All other property functions now accept pressure by default and internally call
``pcsaft_den`` to compute rho, unless ``input='rho'`` is provided.
