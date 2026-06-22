Figures Demo
============

This page demonstrates the ``.. plot::`` directive from
``matplotlib.sphinxext.plot_directive`` for auto-generating
figures at documentation build time.

Ellipsoid iterations
--------------------

.. plot:: examples/plot_ellipsoid_iterations.py

   The ellipsoid search space shrinks as cutting planes are applied.
   The initial (outer) and final (inner) ellipsoids are shown with the
   feasible region shaded in green.

LMI feasibility
---------------

.. plot:: examples/plot_lmi_feasibility.py

   Solving a 2D Linear Matrix Inequality constraint using the ellipsoid
   method. Each snapshot shows the shrinking ellipsoid, with the final
   solution marked by a star.
