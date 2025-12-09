import pyomo.environ as pyo

def solve_multi_period(model, solver_name, solver_options, tee=True):
    """Solve the multi-period ACOPF-with-storage-and-solar model."""
    if solver_options is None:
        solver_options = {"tol": 1e-6}

    solver = pyo.SolverFactory(solver_name)
    for k, v in solver_options.items():
        solver.options[k] = v

    results = solver.solve(model, tee=tee)
    return results

