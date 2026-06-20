---
title: "pysorb: A Python Library and GUI for Adsorption Modelling"
tags:
  - Python
  - adsorption
  - isotherm fitting
  - IAST
  - breakthrough simulation
  - ZLC
  - chemical engineering
authors:
  - name: Hesam Maleki
    orcid: 0000-0003-4639-7739
    affiliation: 1
affiliations:
  - name: Researcher, Cambridge, United Kingdom
    index: 1
date: 16 June 2026
bibliography: paper.bib
---

# Summary

`pysorb` is an open-source Python library and graphical user interface for adsorption modelling. It integrates pure-component isotherm fitting, multicomponent mixture prediction, fixed-bed dynamic breakthrough simulation, Crank diffusion uptake, and zero-length column (ZLC) analysis within a single framework. The same numerical routines are exposed through Python interfaces and Tkinter/Matplotlib graphical workflows, so an analysis can move between scripted research use and interactive laboratory use without changing model definitions or software environment.

The package is written in Python using NumPy [@harris2020array] and SciPy [@virtanen2020scipy], with Numba [@lam2015numba] used to accelerate repeated mixture and breakthrough kernels and Matplotlib for plotting [@hunter2007matplotlib]. The Tkinter and Matplotlib GUI layer supports data import, model configuration, visualisation of fitted curves and breakthrough profiles, result export, and session persistence. The release described here is version `0.2.2`, archived on Zenodo [@maleki2026pysorb].

# Statement of need

Adsorption is a key process in separations, purification, gas storage, and the evaluation of porous materials. Experimental and modelling studies commonly combine equilibrium measurements, which describe how much material is adsorbed under specified thermodynamic conditions, with dynamic and kinetic measurements, which describe how adsorption systems evolve in time. These two classes of information are strongly connected: equilibrium models often provide the local closure relations for dynamic simulations, while kinetic experiments test whether fitted equilibrium and transport parameters are sufficient to describe real adsorption behaviour.

Open-source adsorption modelling has improved substantially over the last two decades. Early dedicated isotherm-fitting software such as ISOFIT introduced hybrid optimisation workflows for sorption data [@matott2008isofit]. Subsequent software expanded mixture prediction, IAST-based calculations, process simulation, isosteric heat analysis, and graphical workflows across Python, MATLAB, and dedicated adsorption simulation codes [@simon2016pyiast; @balestra2016gaiast; @lee2018iastpp; @dautzenberg_graphiast_2022; @ga_pyapep_2023; @kim_dynamic_2023; @li_analytical_2025]. Broader adsorption environments later connected fitting, mixture prediction, adsorption data handling, breakthrough simulation, material characterisation, and GUI-based workflows, as seen in pyGAPS, RUPTURA, and AIM [@iacomi2019pygaps; @sharma2023ruptura; @hassan2025aim]. Together, this literature shows a clear trend from specialised fitting or mixture-prediction tools toward more integrated and accessible adsorption modelling environments. However, there remains a need for a lightweight Python-native platform that brings integrated workflows, accelerated computation, and robust numerical solvers to both scripts and a GUI.

# State of the field

Existing adsorption-modelling tools have made important contributions to fitting, mixture prediction, process simulation, and graphical analysis. However, many available solvers have a narrower feature scope, are designed for a specific part of the modelling chain, use different input conventions, or separate scripted and graphical workflows. These limitations can make complete laboratory workflows fragmented, reducing reproducibility and making numerical accuracy, stability, and failure diagnosis harder to maintain.

`pysorb` is developed as a pure-Python adsorption modelling library and GUI-accessible platform. The current version provides a transparent framework built around robust numerical solvers: an isotherm-fitting solver for pure-component equilibrium data, a mixture solver for unary, Ideal Adsorbed Solution Theory (IAST), and extended mixture calculations, a non-isothermal and non-isobaric breakthrough solver for fixed-bed simulations, and kinetic-analysis solvers for diffusion-controlled uptake and zero-length column experiments. Beyond the core solvers, the platform includes workflow-oriented features, including shared computational routines across Python and GUI interfaces, Numba-accelerated mixture and breakthrough kernels, session handling, plotting, validation checks, convergence information, warning messages, and solver diagnostics. This positions `pysorb` as an integrated adsorption-modelling environment rather than a single-purpose fitting, mixture, or process-simulation utility.

# Software design

`pysorb` is a modular Python code base for adsorption equilibrium and dynamic kinetic modelling. The package depends primarily on NumPy and SciPy, with Numba used in the mixture and breakthrough solvers to compile repeated numerical kernels. The graphical applications use Tkinter and Matplotlib, with spreadsheet import support where needed. At the workflow level, the solvers are organised around pure-component fitting, mixture prediction, fixed-bed breakthrough simulation, Crank diffusion uptake, and zero-length column analysis.

The isotherm-fitting solver estimates pure-component equilibrium parameters from pressure, loading, and temperature data. It supports single-temperature and multi-temperature fitting, with temperature-dependent affinity parameters and isosteric heat calculations. The implemented model families include Linear, Langmuir, Sips, Freundlich, Toth, BET, Quadratic, and Redlich-Peterson forms. Fitting uses bounded nonlinear optimisation with multistart initialisation, user-adjustable bounds, optional constraints, and RMSE, MAE, or average relative error objectives; selected multi-temperature RMSE fits use analytic Jacobians.

The mixture solver supports standalone unary evaluation, Ideal Adsorbed Solution Theory (IAST), and an extended multisite model. Component and site definitions can represent arbitrary numbers of components and sites in the mixture and breakthrough kernels, with mixed site-level model families. The IAST implementation uses Numba-compiled routines, softmax variables for adsorbed-phase mole fractions, log spreading-pressure residuals, and an analytic Jacobian with respect to the softmax variables.

The breakthrough solver implements a one-dimensional, non-isothermal, non-isobaric fixed-bed simulator. The state includes gas-phase mole fractions, adsorbed loadings, bed temperature, wall temperature, and pressure along the column, with equilibrium evaluated by unary, IAST, or extended-mixture kernels. The nonlinear semi-discrete equations are integrated implicitly using Newton iterations, a block-tridiagonal linear solver, line-search damping, retry logic for difficult time steps, and adaptive time-step control.

The Crank diffusion and zero-length column solvers address common laboratory kinetic analyses. The Crank solver calculates spherical Fickian fractional uptake and estimates $D/R^2$ by bounded scalar minimisation. The ZLC solver provides a linear analytical desorption model, plotting tools, and JSON session handling for concentration-decay experiments. Across these solvers, input validation, bounded transformations, parameter sanitisation, finite penalty values, warning mechanisms, convergence flags, residual/error values, and explicit status reporting are used to make numerical failures easier to diagnose.

# Functionality

| Solver | Main purpose | Interface |
| --- | --- | --- |
| Isotherm fitting | Pure-component fitting, temperature-dependent affinities, isosteric heat calculation, and diagnostics | API and GUI |
| Mixture prediction | Unary, IAST, and extended multisite mixture loading calculations for arbitrary component and site counts | API and GUI |
| Breakthrough simulation | Non-isothermal and non-isobaric fixed-bed simulation with coupled mass, heat, and pressure effects | API and GUI |
| Crank diffusion | Spherical Fickian uptake calculation and fitting of $D/R^2$ from uptake data | API and GUI |
| ZLC analysis | Analytical zero-length column desorption calculation, plotting, and session handling | API and GUI |

The graphical workflows provide panels for data import or pasting, model configuration, solver settings, plotting, and result export. Each workflow exposes fitted parameters, calculated profiles, convergence information, or diagnostic output appropriate to the solver. Plots and result tables can be copied for external analysis, and complete GUI sessions can be saved as JSON files. The release includes JSON sessions for a mixture-comparison case, a binary non-isothermal 13X CO~2~/N~2~ breakthrough example, and a five-component isothermal breakthrough example. A notebook example demonstrates scripted use.

# Quality assurance, availability, and reproducibility

`pysorb` is distributed on PyPI and can be installed with:

```bash
pip install pysorb
```

The source code is hosted on GitHub at <https://github.com/hesmalek/pysorb>, and the release associated with this manuscript is archived on Zenodo at <https://doi.org/10.5281/zenodo.20480051>. The software is released under the MIT license. The repository and release materials include installation instructions, documentation, example notebooks, GUI session files, and sample input data.

Quality assurance is supported by automated tests and reproducible examples. The multicomponent IAST example compares `pysorb` loadings with an independent reference calculation for a four-component, two-site-per-component case and reports a maximum absolute deviation of $2.65\times10^{-12}$ for the distributed input. The breakthrough examples exercise the fixed-bed solver on binary non-isothermal and five-component isothermal configurations, including cases that require repeated mixture-equilibrium evaluation during dynamic simulation. These examples give users concrete files for rerunning representative calculations and checking expected behaviour.

# Research impact statement

`pysorb` is intended for researchers who need adsorption calculations that are inspectable, reproducible, and usable in both laboratory and scripted settings. The software contribution is broader than wrapping existing formulas in a GUI: it provides Python-native implementations of the fitting, mixture prediction, breakthrough simulation, and kinetic-analysis workflows, with a shared model representation, numerical safeguards, Numba-accelerated kernels, error handling, diagnostics, and session-based reproducibility built into the user-facing tools.

The immediate research value is in shortening the path from experimental data to reusable models. A user can fit pure-component isotherms, pass the fitted parameters to mixture calculations, use the same equilibrium definitions in a dynamic breakthrough simulation, and analyse uptake or ZLC kinetic measurements in the same environment. This integration is particularly useful for laboratories that need to connect equilibrium and dynamic kinetic analysis without maintaining separate tools or translating parameters between incompatible formats. It supports teaching, internal laboratory workflows, supplementary calculations for publications, and reproducible comparison cases for future adsorption studies.

# AI usage disclosure

LLMs were used for drafting support and language editing while preparing this manuscript. The author reviewed and edited the assisted text, checked the technical claims against the software and examples, and made the scientific, software-design, and submission decisions. The author remains responsible for the accuracy, originality, licensing, and compliance of the submitted materials.

# Acknowledgements

No external funding is declared for this work.
