
.. -*- mode: rst -*-

.. |PythonMinVersion| replace:: 3.7
.. |NumPyMinVersion| replace:: 1.17.3
.. |SciPyMinVersion| replace:: 1.5.0
.. |ScikitLearnMinVersion| replace:: 1.0.2
.. |DESlibMinVersion| replace:: 0.3


Discriminate index (DISi)
=========================

DISi is an implementation techniques for dynamic classifier and ensemble selection.
The library is is based on scikit-learn_ and deslib_.


Dependencies:
-------------

The dependency requirements are:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- Scikit-learn (>= |ScikitLearnMinVersion|)
- DESlib (>= |DESlibMinVersion|)


Examples:
---------

Here we show an example:

.. code-block:: python

    from discr_idx import DISi

    # Base estimator: perceptron
    estimator = Perceptron(max_iter=100)

    # Train a pool
    pool_classifiers = BaggingClassifier(estimator).fit(xtrain, ytrain)

    # Initialize DISi
    disi = DISi(pool_classifiers).fit(xdsel, ydsel)

    # Predict xtest
    disi.predict(xtest)


Citation
---------

If you use Discriminate index (DISi), please consider citing the following paper:

Marcelo R. Trierveiler, Alceu S. Britto, Luiz Oliveira and Robert Sabourin `Dynamic Ensemble Selection by K-Nearest Local Oracles with Discrimination Index <https://doi.ieeecomputersociety.org/10.1109/ICTAI.2018.00120>`_ In: 2018 IEEE 30th ICTAI. IEEE Computer Society (2018). 


.. code-block:: text

    @inproceedings{TRIER2018a:discrimindex,
        author = {Marcelo R. Trierveiler and Alceu S. Britto and Luiz Oliveira and Robert Sabourin},
        title = {Dynamic Ensemble Selection by K-Nearest Local Oracles with Discrimination Index},
        booktitle = {2018 IEEE 30th International Conference on Tools with Artificial Intelligence (ICTAI)},
        year = {2018},
        doi = {10.1109/ICTAI.2018.00120},
        publisher = {IEEE Computer Society},
        isbn = {978-1-5386-7449-9},
        issn = {2375-0197},
        url = {https://doi.ieeecomputersociety.org/10.1109/ICTAI.2018.00120}
    }


References:
-----------

.. [1] : R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,” Information Fusion, vol. 41, pp. 195 – 216, 2018.

.. [2] : R. M. O. Cruz, L. G. Hafemann, R. Sabourin and G. D. C. Cavalcanti “DESlib: A Dynamic ensemble selection library in Python,” Journal of Machine Learning Research, vol. 21, pp. 1 – 5, 2020.

.. _scikit-learn: http://scikit-learn.org/stable/

.. _numpy: http://www.numpy.org/

.. _deslib: https://github.com/scikit-learn-contrib/DESlib
