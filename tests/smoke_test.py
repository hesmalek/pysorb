import numpy as np

import pysorb
from pysorb.crank import crank_uptake, fit_D_R2
from pysorb.fit import fit_isotherm
from pysorb.mix import ext, iast, parse, unary
from pysorb.zlc import linmodel


def main():
    print("pysorb", pysorb.__version__)

    q_crank = crank_uptake([0, 1, 2], D_R2=0.001)
    fitted_d_r2, fit_error = fit_D_R2([0, 1, 2], [0, 0.1, 0.14])
    print("crank", np.round(q_crank, 8).tolist(), round(float(fitted_d_r2), 8), round(float(fit_error), 8))

    zlc_result = linmodel(20, 6.4e-3, 0.001, 50, 2, NN=5, dt=1, return_N=True)
    print("zlc", len(zlc_result), len(zlc_result[0]), round(float(zlc_result[5]), 6), zlc_result[-1])

    components = [
        {"MoleculeName": "A", "isotherms": [["Langmuir", 1.0, 0.5]]},
        {"MoleculeName": "B", "isotherms": [["Langmuir", 2.0, 0.2]]},
    ]
    parsed = parse(components)
    print("mix unary", np.round(unary([1.0, 0.5], parsed=parsed), 8).tolist())
    print("mix ext", np.round(ext([1.0, 0.5], parsed=parsed), 8).tolist())
    print("mix iast", np.round(iast([1.0, 0.5], parsed=parsed, max_iter=20), 8).tolist())

    P = np.array([0.1, 0.3, 0.7, 1.2, 2.0])
    T = np.full_like(P, 298.15)
    q = 1.5 * 0.8 * P / (1 + 0.8 * P)
    fit = fit_isotherm(P, q, T, "Langmuir", mode="single", n_trials=2, max_nfev=1000, random_state=1)
    print("fit", fit.model_name, fit.mode, np.round(fit.parameters, 8).tolist(), round(float(fit.metrics["RMSE"]), 10))

    from pysorb.bt import run_simulation

    print("bt", callable(run_simulation))


if __name__ == "__main__":
    main()
