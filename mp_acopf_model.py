import json
import math
import pyomo.environ as pyo

from egret.parsers.matpower_parser import create_ModelData
from egret.models import acopf

def build_multi_period_acopf_with_storage(
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
    Multi-period ACOPF using Egret ACOPF as sub-blocks with:
      * Time-varying active loads per bus (from JSON, in MW)
      * One continuous storage device at 'storage_bus'
      * One solar generator at 'solar_bus' with base capacity 'solar_capacity_MW'
        and time-varying availability 'solar_profile'.

    Storage model (all continuous, no binaries):
      e_t        [0, E_max]      (energy storage level of the battery, pu·h)
      p_ch_t     [0, P_ch_max]   (charge power or power input to battery, pu)
      p_dis_t    [0, P_dis_max]  (discharge power or power output of battery, pu)
      e_t = e_{t-1} + eta_ch * dt * p_ch_t - (dt / eta_dis) * p_dis_t
      Nodal balance at storage bus b:
          (original Nodal balance) + p_dis_t - p_ch_t = 0

    Solar model:
      - A generator 'solar_gen_name' is created at 'solar_bus'
        with base p_max = 'solar_capacity_MW' (MW).
      - For each period t:
            p_max_t = solar_capacity_MW * solar_profile[t_idx]
      - Solar variable cost is forced to zero by zeroing its p_cost coefficients!!
    """

    # Read load profile, a json file (MW)
    with open(load_profile_path, "r") as f:
        load_data = json.load(f)
    load_MW_raw = load_data["load_MW"] 

    # Load base ModelData and set up baseMVA, buses, generators
    base_md = create_ModelData(matpower_case_path)

    system_dict = base_md.data.get("system", {})
    base_mva = (
        system_dict.get("base_mva")
        or system_dict.get("baseMVA")
        or 100.0  
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

    # Create a solar generator with a given capacity
    if solar_gen_name not in gens:
        # Use first existing generator as a template 
        template_name, template_gen = next(iter(gens.items()))
        new_gen = template_gen.copy()

        # Put solar at chosen bus, with desired capacity
        new_gen["bus"] = solar_bus          
        new_gen["p_min"] = 0.0
        new_gen["p_max"] = solar_capacity_MW  # base capacity in MW
        new_gen["q_min"] = template_gen.get("q_min", -9999) # I'm a bit skeptical about these defaults. Solar PV produce reactive power?
        new_gen["q_max"] = template_gen.get("q_max", 9999)
        new_gen["in_service"] = True
        base_md.data["elements"]["generator"][solar_gen_name] = new_gen
        gens = dict(base_md.elements(element_type="generator"))

    solar_gen = base_md.data["elements"]["generator"][solar_gen_name]

    # Force solar variable cost to zero 
    pc = solar_gen.get("p_cost", None)
    # Seems like the Egret structure pc["values"] is a dict of {power: coeff}? Double check...
    if isinstance(pc, dict) and isinstance(pc.get("values"), dict):
        for k in pc["values"]:
            pc["values"][k] = 0.0
        solar_gen["p_cost"] = pc

    base_md.data["elements"]["generator"][solar_gen_name] = solar_gen

    # Create a random solar availability profile. This could be a data input too...
    T_list = list(time_periods)
    n_T = len(T_list)

    # Check load profile length just in case.
    for bus_key, series in load_MW_raw.items():
        if len(series) != n_T:
            raise ValueError(
                f"Load profile for bus {bus_key} has length {len(series)}, "
                f"but time_periods has length {n_T}."
            )

    # If no solar_profile given, generate a sort of bell-shaped curve
    if solar_profile is None:
        solar_profile = []
        for idx in range(n_T):
            hour = idx  # assume t=1..24 ... idx=0..23
            x = (hour - 12) / 6.0
            cf = math.exp(-x * x)
            solar_profile.append(cf)
        max_cf = max(solar_profile) or 1.0
        solar_profile = [cf / max_cf for cf in solar_profile]
        # now, make the solar availability = 0 in times that are not 7,8,9,...,17,18
        for idx in range(n_T):
            hour = idx
            if hour < 7 or hour > 18:
                solar_profile[idx] = 0.0
    
    if len(solar_profile) != n_T:
        raise ValueError(
            f"solar_profile length {len(solar_profile)} != number of periods {n_T}"
        )

    # Create a top-level multi-period model. The period is from 1 to 24. could be more.
    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=T_list, ordered=True)

    # ACOPF block per period: time-varying loads + solar capacity
    def _opf_block_rule(block, t):
        # Clone base ModelData (with solar with zero cost)
        md_t = base_md.clone()
        t_idx = T_list.index(t)

        # Overwrite loads for period t (MW), very important. 
        loads_t = dict(md_t.elements(element_type="load"))
        for load_name, load in loads_t.items():
            bus = load["bus"]
            bus_key = str(bus)
            if bus_key in load_MW_raw:
                load["p_load"] = load_MW_raw[bus_key][t_idx] * 0.2

        # Overwrite solar p_max for period t, very important 
        gens_t = dict(md_t.elements(element_type="generator"))
        solar_gen_t = gens_t[solar_gen_name]
        cf_t = solar_profile[t_idx]  # 0..1
        solar_gen_t["p_min"] = 0.0
        solar_gen_t["p_max"] = solar_capacity_MW * cf_t
        md_t.data["elements"]["generator"][solar_gen_name] = solar_gen_t

        # Build Egret ACOPF model for this period 
        opf_model_t, md_scaled_t = acopf.create_psv_acopf_model(
            md_t,
            include_feasibility_slack=include_feasibility_slack,
        )

        block._md = md_scaled_t
        block.transfer_attributes_from(opf_model_t)

    m.OPF = pyo.Block(m.T, rule=_opf_block_rule)

    # Storage variables (continuous, no cost)
    m.e = pyo.Var(m.T, bounds=(0, E_max_pu))          # SoC or energy level of battery [pu·h]
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
        # this constraint might not be necessary. charging and discharging at the same time might not be a problem?
        return m.p_ch[t] * m.p_dis[t] <= 1e-6
    
    m.NoSimultaneous = pyo.Constraint(m.T, rule=no_simultaneous_ch_dis_rule)

    # Modify nodal balance at storage bus: add p_dis - p_ch
    # should modify at EACH BUS THAT CONTAINS A BATTERY but this example has only one
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

    # Objective: sum of hourly generator operating costs
    # solar has zero p_cost (by construction above)
    # storage has no direct cost here
    for t in m.T:
        m.OPF[t].obj.deactivate()

    def total_cost_rule(m):
        return sum(m.OPF[t].obj.expr for t in m.T)

    m.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    return m, base_md
