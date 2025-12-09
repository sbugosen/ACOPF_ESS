import pyomo.environ as pyo
from idaes.core.solvers import get_solver

from mp_acopf_model import build_multi_period_acopf_with_storage 
from mp_acopf_solver import solve_multi_period

if __name__ == "__main__":
    CASE_PATH = "data/pglib_opf_case14_ieee.m"
    LOAD_PROFILE_PATH = "data/load_case_14_2026-01-01.json"

    T = range(1, 25)
    STORAGE_BUS_ID = "3"
    SOLAR_BUS_ID = "5"
    SOLAR_GEN_NAME = "Gen_Solar"
    SOLAR_CAPACITY_MW = 40.0

    E_max_MWh = 50.0
    P_ch_max_MW = 25.0
    P_dis_max_MW = 25.0

    model, base_md = build_multi_period_acopf_with_storage(
        matpower_case_path=CASE_PATH,
        load_profile_path=LOAD_PROFILE_PATH,
        time_periods=T,
        storage_bus=STORAGE_BUS_ID,
        E_max_MWh=E_max_MWh,
        P_ch_max_MW=P_ch_max_MW,
        P_dis_max_MW=P_dis_max_MW,
        solar_gen_name=SOLAR_GEN_NAME,
        solar_bus=SOLAR_BUS_ID,
        solar_capacity_MW=SOLAR_CAPACITY_MW,
        solar_profile=None,
        eta_ch=0.95,
        eta_dis=0.95,
        dt_hours=1.0,
        include_feasibility_slack=False,
    )

    results = solve_multi_period(
        model,
        solver_name=get_solver("ipopt").name,
        solver_options={"tol": 1e-7},
        tee=True,
    )

    print("\nTotal cost:", pyo.value(model.TotalCost))

