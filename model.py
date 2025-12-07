import json
import math
import pyomo.environ as pyo

from egret.parsers.matpower_parser import create_ModelData
from egret.models import acopf
from idaes.core.solvers import get_solver

def build_multi_period_acopf_with_storage_solar_and_load(
    matpower_case_path: str,
    load_profile_path: str,
    time_periods,
    storage_bus,
    E_max_MWh: float,
    P_ch_max_MW: float,
    P_dis_max_MW: float,
    solar_gen_name: str,
    solar_bus,
    solar_capacity_MW: float,
    solar_profile=None,
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
    dt_hours: float = 1.0,
    include_feasibility_slack: bool = False,
):
    """
    Multi-period ACOPF using Egret ACOPF (PSV) as sub-blocks with:
      * Time-varying active loads per bus (from JSON, in MW)
      * One continuous storage device at `storage_bus`
      * One solar generator at `solar_bus` with base capacity `solar_capacity_MW`
        and time-varying availability `solar_profile` ∈ [0,1]^T.

    Storage model (all continuous, no binaries):
      e_t      ∈ [0, E_max]      (SoC, pu·h)
      p_ch_t   ∈ [0, P_ch_max]   (charge power, pu)
      p_dis_t  ∈ [0, P_dis_max]  (discharge power, pu)
      e_t = e_{t-1} + η_ch * dt * p_ch_t - (dt / η_dis) * p_dis_t
      P-balance at storage bus b*:
          (original P-balance) + p_dis_t - p_ch_t = 0

    Solar model:
      - A generator `solar_gen_name` is created (if missing) at `solar_bus`
        with base p_max = `solar_capacity_MW` (MW).
      - For each period t:
            p_max_t = solar_capacity_MW * solar_profile[t_idx]
      - Solar variable cost is forced to zero by zeroing its p_cost coefficients.
    """

    # ------------------------------------------------------------------
    # 0) Read load profile from JSON (MW)
    # ------------------------------------------------------------------
    with open(load_profile_path, "r") as f:
        load_data = json.load(f)
    load_MW_raw = load_data["load_MW"]  # dict: bus(str) -> list

    # ------------------------------------------------------------------
    # 1) Load base ModelData and set up baseMVA, buses, generators
    # ------------------------------------------------------------------
    base_md = create_ModelData(matpower_case_path)

    system_dict = base_md.data.get("system", {})
    base_mva = (
        system_dict.get("base_mva")
        or system_dict.get("baseMVA")
        or 100.0  # safe default
    )

    # Convert storage params to pu
    E_max_pu = E_max_MWh / base_mva
    P_ch_max_pu = P_ch_max_MW / base_mva
    P_dis_max_pu = P_dis_max_MW / base_mva

    storage_bus_id = str(storage_bus)
    solar_bus_id = str(solar_bus)

    buses = dict(base_md.elements(element_type="bus"))
    if storage_bus_id not in buses:
        raise KeyError(
            f"storage_bus {storage_bus!r} not found in case buses: {list(buses.keys())}"
        )
    if solar_bus_id not in buses:
        raise KeyError(
            f"solar_bus {solar_bus!r} not found in case buses: {list(buses.keys())}"
        )

    gens = dict(base_md.elements(element_type="generator"))

    # ------------------------------------------------------------------
    # 2) Ensure solar generator exists: if not, create it with given capacity
    # ------------------------------------------------------------------

    if solar_gen_name not in gens:
        # Use first existing generator as a template for misc fields / Q limits / cost structure
        template_name, template_gen = next(iter(gens.items()))
        new_gen = template_gen.copy()

        # Put solar at chosen bus, with desired capacity
        new_gen["bus"] = solar_bus          # as in case (int or str)
        new_gen["p_min"] = 0.0
        new_gen["p_max"] = solar_capacity_MW  # base capacity in MW
        new_gen["q_min"] = template_gen.get("q_min", -9999)
        new_gen["q_max"] = template_gen.get("q_max", 9999)
        new_gen["in_service"] = True
        base_md.data["elements"]["generator"][solar_gen_name] = new_gen
        gens = dict(base_md.elements(element_type="generator"))

    # Refresh solar_gen from base_md
    solar_gen = base_md.data["elements"]["generator"][solar_gen_name]

    # ------------------------------------------------------------------
    # 2b) Force solar variable cost to zero WITHOUT changing structure
    # ------------------------------------------------------------------
    pc = solar_gen.get("p_cost", None)
    # Common Egret structure: pc["values"] is a dict of {power: coeff} or {order: coeff}
    if isinstance(pc, dict) and isinstance(pc.get("values"), dict):
        for k in pc["values"]:
            pc["values"][k] = 0.0
        solar_gen["p_cost"] = pc
    # If structure is something else, just leave it as-is for now (won't break scaling)

    base_md.data["elements"]["generator"][solar_gen_name] = solar_gen

    # ------------------------------------------------------------------
    # 3) Solar availability profile
    # ------------------------------------------------------------------
    T_list = list(time_periods)
    n_T = len(T_list)

    # Check load profile length
    for bus_key, series in load_MW_raw.items():
        if len(series) != n_T:
            raise ValueError(
                f"Load profile for bus {bus_key} has length {len(series)}, "
                f"but time_periods has length {n_T}."
            )

    # If no solar_profile given, generate a smooth bell-shaped curve
    if solar_profile is None:
        solar_profile = []
        for idx in range(n_T):
            hour = idx  # assume t=1..24 ↔ idx=0..23
            x = (hour - 12) / 6.0
            cf = math.exp(-x * x)
            solar_profile.append(cf)
        max_cf = max(solar_profile) or 1.0
        solar_profile = [cf / max_cf for cf in solar_profile]

    if len(solar_profile) != n_T:
        raise ValueError(
            f"solar_profile length {len(solar_profile)} != number of periods {n_T}"
        )

    # ------------------------------------------------------------------
    # 4) Top-level multi-period model
    # ------------------------------------------------------------------
    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=T_list, ordered=True)

    # ------------------------------------------------------------------
    # 5) ACOPF block per period: time-varying loads + solar capacity
    # ------------------------------------------------------------------
    def _opf_block_rule(block, t):
        # Clone base ModelData (with solar & zero cost already baked in)
        md_t = base_md.clone()
        t_idx = T_list.index(t)

        # ---- Overwrite loads for period t (MW) ----
        loads_t = dict(md_t.elements(element_type="load"))
        for load_name, load in loads_t.items():
            bus = load["bus"]
            bus_key = str(bus)
            if bus_key in load_MW_raw:
                load["p_load"] = load_MW_raw[bus_key][t_idx] * 0.2

        # ---- Overwrite solar p_max for period t ----
        gens_t = dict(md_t.elements(element_type="generator"))
        solar_gen_t = gens_t[solar_gen_name]
        cf_t = solar_profile[t_idx]  # 0..1
        solar_gen_t["p_min"] = 0.0
        solar_gen_t["p_max"] = solar_capacity_MW * cf_t
        md_t.data["elements"]["generator"][solar_gen_name] = solar_gen_t

        # ---- Build Egret ACOPF model for this period ----
        opf_model_t, md_scaled_t = acopf.create_psv_acopf_model(
            md_t,
            include_feasibility_slack=include_feasibility_slack,
        )

        block._md = md_scaled_t
        block.transfer_attributes_from(opf_model_t)

    m.OPF = pyo.Block(m.T, rule=_opf_block_rule)

    # ------------------------------------------------------------------
    # 6) Storage variables and dynamics (continuous, no cost)
    # ------------------------------------------------------------------
    m.e = pyo.Var(m.T, bounds=(0, E_max_pu))          # SoC [pu·h]
    m.p_ch = pyo.Var(m.T, bounds=(0, P_ch_max_pu))    # charge [pu]
    m.p_dis = pyo.Var(m.T, bounds=(0, P_dis_max_pu))  # discharge [pu]

    e_init_pu = 0.0  # initial SoC (pu·h)

    def soc_rule(m, t):
        if t == m.T.first():
            return m.e[t] == e_init_pu + eta_ch * dt_hours * m.p_ch[t] \
                   - (dt_hours / eta_dis) * m.p_dis[t]
        t_prev = m.T.prev(t)
        return m.e[t] == m.e[t_prev] + eta_ch * dt_hours * m.p_ch[t] \
               - (dt_hours / eta_dis) * m.p_dis[t]

    m.SoC = pyo.Constraint(m.T, rule=soc_rule)

    def no_simultaneous_ch_dis_rule(m, t):
        # forbid simultaneous charge and discharge (approximate complementarity)
        return m.p_ch[t] * m.p_dis[t] <= 1e-6
    
    m.NoSimultaneous = pyo.Constraint(m.T, rule=no_simultaneous_ch_dis_rule)

    # ------------------------------------------------------------------
    # 7) Modify P-balance at storage bus: add p_dis - p_ch
    # ------------------------------------------------------------------
    for t in m.T:
        sub = m.OPF[t]
        if storage_bus_id not in sub.eq_p_balance:
            raise KeyError(
                f"Bus {storage_bus_id!r} not present in eq_p_balance for t={t}."
            )
        sub.eq_p_balance[storage_bus_id].deactivate()

    def p_balance_with_storage_rule(m, t):
        sub = m.OPF[t]
        original_body = sub.eq_p_balance[storage_bus_id].body
        return original_body + m.p_dis[t] - m.p_ch[t] == 0.0

    m.eq_p_balance_storage = pyo.Constraint(m.T, rule=p_balance_with_storage_rule)

    # ------------------------------------------------------------------
    # 8) Objective: sum of hourly generator operating costs
    #     - solar has zero p_cost (by construction above)
    #     - storage has no direct cost here
    # ------------------------------------------------------------------
    for t in m.T:
        m.OPF[t].obj.deactivate()

    def total_cost_rule(m):
        return sum(m.OPF[t].obj.expr for t in m.T)

    m.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    return m, base_md

def solve_multi_period(
    model,
    solver_name: str = "ipopt",
    solver_options: dict | None = None,
    tee: bool = True,
):
    """Solve the multi-period ACOPF-with-storage-and-solar model."""
    if solver_options is None:
        solver_options = {"tol": 1e-6}

    solver = pyo.SolverFactory(solver_name)
    for k, v in solver_options.items():
        solver.options[k] = v

    results = solver.solve(model, tee=tee)
    return results


def print_storage_and_solar_summary(model, base_md):
    """Print SoC and storage power in MW, plus solar Pg for each period."""
    system_dict = base_md.data.get("system", {})
    base_mva = (
        system_dict.get("base_mva")
        or system_dict.get("baseMVA")
        or 100.0
    )

    print("\n=== STORAGE SCHEDULE (MW/MWh) ===")
    for t in model.T:
        e_MWh = pyo.value(model.e[t]) * base_mva
        p_ch_MW = pyo.value(model.p_ch[t]) * base_mva
        p_dis_MW = pyo.value(model.p_dis[t]) * base_mva
        print(
            f"t={t:2d}: "
            f"SoC={e_MWh:7.2f} MWh, "
            f"P_ch={p_ch_MW:7.2f} MW, "
            f"P_dis={p_dis_MW:7.2f} MW"
        )

    # Optionally, also show solar dispatch per period if you like:
    print("\n=== SAMPLE GENERATOR DISPATCH (PU) FOR EACH t ===")
    for t in model.T:
        print(f"\n  t = {t}")
        opf = model.OPF[t]
        for g in opf.pg:
            pg = pyo.value(opf.pg[g])
            print(f"    Gen {g}: Pg = {pg:7.4f} pu")


import math
import pyomo.environ as pyo


def print_full_acopf_solution(model, base_md, print_branches=True):
    """
    Print generator outputs, bus voltages/angles, and (optionally) branch flows
    for each time period in the multi-period ACOPF+storage model.

    All powers are reported in MW / Mvar (converted from pu with baseMVA).
    Voltages are in p.u., angles in degrees.
    """
    system_dict = base_md.data.get("system", {})
    base_mva = (
        system_dict.get("base_mva")
        or system_dict.get("baseMVA")
        or 100.0
    )

    print("\n==================== ACOPF SOLUTION BY TIME PERIOD ====================\n")

    for t in model.T:
        opf = model.OPF[t]
        print(f"==================== t = {t} ====================")

        # -------- Generators --------
        print("\nGenerators (Pg, Qg):")
        if hasattr(opf, "pg") and hasattr(opf, "qg"):
            for g in sorted(opf.pg.index_set()):
                pg_MW = pyo.value(opf.pg[g]) * base_mva
                qg_Mvar = pyo.value(opf.qg[g]) * base_mva
                print(f"  Gen {g:>6}: Pg = {pg_MW:8.2f} MW,  Qg = {qg_Mvar:8.2f} Mvar")
        else:
            print("  (No pg/qg variables found in this OPF block)")

        # -------- Buses --------
        print("\nBuses (Vm, Va):")
        # PSV ACOPF has vm, va
        if hasattr(opf, "vm") and hasattr(opf, "va"):
            for b in sorted(opf.vm.index_set()):
                vm = pyo.value(opf.vm[b])      # p.u.
                va_deg = math.degrees(pyo.value(opf.va[b]))  # rad → deg
                print(f"  Bus {b:>4}: Vm = {vm:6.3f} p.u.,  Va = {va_deg:8.3f} deg")
        else:
            print("  (No vm/va variables found in this OPF block)")

        # -------- Branches --------
        if print_branches and hasattr(opf, "pf"):
            print("\nBranches (Pf, Qf, Pt, Qt):")
            # pf, qf: from side; pt, qt: to side
            for br in sorted(opf.pf.index_set()):
                pf_MW = pyo.value(opf.pf[br]) * base_mva
                qf_Mvar = pyo.value(opf.qf[br]) * base_mva
                pt_MW = pyo.value(opf.pt[br]) * base_mva
                qt_Mvar = pyo.value(opf.qt[br]) * base_mva
                print(
                    f"  Branch {br:>6}: "
                    f"Pf = {pf_MW:8.2f} MW,  Qf = {qf_Mvar:8.2f} Mvar,  "
                    f"Pt = {pt_MW:8.2f} MW,  Qt = {qt_Mvar:8.2f} Mvar"
                )

        print()  # blank line between periods



import matplotlib.pyplot as plt
import numpy as np

def extract_profiles(model, base_md, solar_gen_name="Gen_Solar"):
    """Extract hourly profiles: thermal gen, solar gen, load, storage."""
    system_dict = base_md.data.get("system", {})
    base_mva = (
        system_dict.get("base_mva")
        or system_dict.get("baseMVA")
        or 100.0
    )

    hours = sorted(list(model.T))
    nT = len(hours)

    thermal_MW = np.zeros(nT)
    solar_MW   = np.zeros(nT)
    load_MW    = np.zeros(nT)
    soc_MWh    = np.zeros(nT)
    p_ch_MW    = np.zeros(nT)
    p_dis_MW   = np.zeros(nT)

    for i, t in enumerate(hours):
        opf = model.OPF[t]

        # Generation
        for g in opf.pg:
            pg_MW = pyo.value(opf.pg[g]) * base_mva
            if g == solar_gen_name:
                solar_MW[i] += pg_MW
            else:
                thermal_MW[i] += pg_MW

        # Load: sum of fixed pl at all buses
        if hasattr(opf, "pl"):
            total_load_pu = sum(pyo.value(opf.pl[b]) for b in opf.pl)
            load_MW[i] = total_load_pu * base_mva

        # Storage
        soc_MWh[i]  = pyo.value(model.e[t]) * base_mva
        p_ch_MW[i]  = pyo.value(model.p_ch[t]) * base_mva
        p_dis_MW[i] = pyo.value(model.p_dis[t]) * base_mva

    return {
        "hours": hours,
        "thermal_MW": thermal_MW,
        "solar_MW": solar_MW,
        "load_MW": load_MW,
        "soc_MWh": soc_MWh,
        "p_ch_MW": p_ch_MW,
        "p_dis_MW": p_dis_MW,
    }


#def plot_power_profiles(profiles):
#    hours      = profiles["hours"]
#    thermal_MW = profiles["thermal_MW"]
#    solar_MW   = profiles["solar_MW"]
#    load_MW    = profiles["load_MW"]
#
#    x = np.array(hours)
#
#    plt.figure(figsize=(10, 5))
#    # stacked bars: thermal + solar
#    plt.bar(x, thermal_MW, label="Generator", color="navy")
#    plt.bar(x, solar_MW, bottom=thermal_MW, label="Solar farm", color="tomato")
#
#    # load as a line
#    plt.plot(x, load_MW, "-o", label="Load", color="limegreen")
#
#    plt.xlabel("Hour")
#    plt.ylabel("MW")
#    plt.title("Power profiles with storage unit")
#    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
#    plt.legend()
#    plt.tight_layout()
#    plt.show()


def plot_storage_profiles(profiles):
    hours    = profiles["hours"]
    soc_MWh  = profiles["soc_MWh"]
    p_ch_MW  = profiles["p_ch_MW"]
    p_dis_MW = profiles["p_dis_MW"]

    x = np.array(hours)

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.step(x, soc_MWh, where="post")
    plt.ylabel("SoC [MWh]")
    plt.title("Storage schedule")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.bar(x - 0.15, p_ch_MW, width=0.3, label="Charge", color="tab:blue")
    plt.bar(x + 0.15, p_dis_MW, width=0.3, label="Discharge", color="tab:orange")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_power_profiles(profiles):
    hours      = profiles["hours"]
    thermal_MW = profiles["thermal_MW"]
    solar_MW   = profiles["solar_MW"]
    load_MW    = profiles["load_MW"]
    p_ch_MW    = profiles["p_ch_MW"]
    p_dis_MW   = profiles["p_dis_MW"]

    x = np.array(hours)

    # net demand seen by generators (approximate, ignores losses)
    net_demand = load_MW + p_ch_MW 

    plt.figure(figsize=(10, 5))
    plt.bar(x, thermal_MW, label="Generator", color="navy")
    plt.bar(x, solar_MW, bottom=thermal_MW, label="Solar farm", color="tomato")
    plt.bar(x, p_dis_MW, bottom=thermal_MW + solar_MW, label="Storage discharge", color="gold")

    plt.plot(x, net_demand, "-o", label="Net demand", color="limegreen")

    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title("Power profiles with storage unit")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def check_power_balance(profiles):
    hours      = profiles["hours"]
    thermal    = profiles["thermal_MW"]
    solar      = profiles["solar_MW"]
    load       = profiles["load_MW"]
    p_ch       = profiles["p_ch_MW"]
    p_dis      = profiles["p_dis_MW"]

    print("\n=== POWER BALANCE CHECK (MW) ===")
    print("t  Gen(therm+sol+dis)   Load    p_ch   approx_losses")
    for i, t in enumerate(hours):
        gen_total = thermal[i] + solar[i] + p_dis[i]
        rhs = load[i] + p_ch[i]
        losses = gen_total - rhs
        print(
            f"{t:2d}   {gen_total:10.2f}   {load[i]:7.2f}  {p_ch[i]:7.2f}   {losses:10.2f}"
        )


if __name__ == "__main__":
    CASE_PATH = "pglib_opf_case14_ieee.m"
    LOAD_PROFILE_PATH = "load_case_14_2026-01-01.json"
    
    T = range(1, 25)
    STORAGE_BUS_ID = '3'
    SOLAR_BUS_ID = '5'
    SOLAR_GEN_NAME = "Gen_Solar"   # this is fine now; code will create it
    SOLAR_CAPACITY_MW = 40.0       # base solar size you want
    
    E_max_MWh = 50.0
    P_ch_max_MW = 25.0
    P_dis_max_MW = 25.0
    
    model, base_md = build_multi_period_acopf_with_storage_solar_and_load(
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
        solver_name = get_solver("ipopt").name,
        solver_options={"tol": 1e-7},
        tee=True,
    )

    print("\nTotal cost (pu-cost units):", pyo.value(model.TotalCost))
    print_storage_and_solar_summary(model, base_md)
    #print_full_acopf_solution(model, base_md)

    profiles = extract_profiles(model, base_md, solar_gen_name=SOLAR_GEN_NAME)
    check_power_balance(profiles)
    plot_power_profiles(profiles)
    plot_storage_profiles(profiles)
