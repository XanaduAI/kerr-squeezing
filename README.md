# Kerr Squeezing in Waveguides

This repository contains the source code used to produce the numerical results presented in
*"Degenerate Squeezing in Waveguides: A Unified Theoretical Approach"* [Journal of Physics: Photonics *2* 035001 (2020)](https://doi.org/10.1088/2515-7647/ab87fc)

## Contents

The library `kerrlib.py` uses the methods derived in the reference above to numerically propagate the mean field and quadratic moments of a pulse undergoing a cubic nonlinearity in a waveguide. `test_kerrlib.py` contains unit tests of the main library. Finally, in `numerical_example_dual_pump.py` you can find an example calculation, described in detail in Sec. III.C. of the paper above.

## Requirements

To perform numerical evaluations this library requires [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/). The unit tests use [pytest](https://docs.pytest.org/en/latest/), [matplotlib](https://matplotlib.org/) is used for graph generation and the `decompositions` module of [Strawberry Fields](https://strawberryfields.readthedocs.io/en/latest/) is used to perform Autonne decompositions in `plot_gen.py`.

All of these prerequisites can be installed via `pip`:

```bash
pip install numpy scipy pytest matplotlib strawberryfields
```

## Authors

L.G. Helt and N. Quesada

If you are doing any research using this source code, please cite the following paper:

> L.G. Helt and N. Quesada. *"Degenerate Squeezing in Waveguides: A Unified Theoretical Approach"* [Journal of Physics: Photonics *2* 035001 (2020)](https://doi.org/10.1088/2515-7647/ab87fc)

## License

This source code is free and open source, released under the Apache License, Version 2.0.
