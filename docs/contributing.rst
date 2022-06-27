Contributing Guide
==================

We are happy to receive any contributions! Please review the following material to learn
how to contribute.

If you are new to GitHub or git, you may find `this guide
<https://docs.github.com/en/get-started/using-git/about-git#basic-git>`_ useful.

Setting up a development environment
------------------------------------

First, fork the repository on GitHub, then clone your forked repository.

.. code-block:: bash

    git clone https://github.com/your-username/BaSiCPy.git
    cd BaSiCPy

Then, create a virtual environment and install the devlopment version of the package.

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate
    pip install -e '.[dev]'

Install pre-commit to check code formatting.

.. code-block:: bash

    pre-commit install
