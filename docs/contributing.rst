Contributing
============

We welcome contributions! Here's how you can help:

Setting Up Development Environment
-----------------------------------

1. Fork the repository
2. Clone your fork:

.. code-block:: bash

    git clone https://github.com/yourusername/taco.git
    cd taco

3. Install development dependencies:

.. code-block:: bash

    pip install -e ".[dev,docs]"
    pre-commit install

Running Tests
-------------

.. code-block:: bash

    pytest

Running Linters
---------------

.. code-block:: bash

    ruff check src tests
    black src tests
    mypy src

Code Style
----------

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions small and focused

Pull Request Process
--------------------

1. Create a new branch for your feature
2. Make your changes
3. Add tests
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request
