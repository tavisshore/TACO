Installation
============

From PyPI
---------

.. code-block:: bash

    pip install taco

From Source
-----------

.. code-block:: bash

    git clone https://github.com/yourusername/taco.git
    cd taco
    pip install -e ".[dev]"

Development Setup
-----------------

To set up the development environment:

.. code-block:: bash

    # Install dependencies
    pip install -e ".[dev,docs]"

    # Install pre-commit hooks
    pre-commit install

    # Run tests
    pytest

    # Run linters
    ruff check src tests
    black src tests
    mypy src
