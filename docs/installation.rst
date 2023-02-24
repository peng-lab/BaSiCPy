Installation
============

For Mac (Intel chip), Linux or WSL2 users
-----------------------------------------

Install from PyPI

.. code-block:: bash

   pip install basicpy

or install the latest development version

.. code-block:: bash

   git clone https://github.com/peng-lab/BaSiCPy.git
   cd BaSiCPy
   pip install .


For Mac users with M1 chip
--------------------------

BaSiCPy requires `jax <https://github.com/google/jax/>`_,
which has potential build issue with M1 chips.
One easiest solution is using `Miniforge <https://github.com/conda-forge/miniforge>`_
as explained `here <https://github.com/google/jax/issues/5501>`_.
In the Miniforge environment, please try the following:

.. code-block:: bash

   pip install "jax[cpu]==0.3.22" jaxlib
   pip install basicpy

For Windows users
-----------------

BaSiCPy requires `jax <https://github.com/google/jax/>`_ which does not support Windows officially.
However, thanks to `cloudhan/jax-windows-builder <https://github.com/cloudhan/jax-windows-builder>`_, we can install BaSiCPy as follows:

.. code-block:: bash

   pip install "jax[cpu]==0.3.14" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
   pip install basicpy

For details and latest updates, see `this issue <https://github.com/google/jax/issues/438>`_.

Install with dev dependencies
-----------------------------

One can use `venv` as:

.. code-block:: bash

   git clone https://github.com/peng-lab/BaSiCPy.git
   cd BaSiCPy
   python -m venv venv
   source venv/bin/activate
   pip install -e '.[dev]'
