import os
import random
import math
import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*GDAL.*')
import geopandas as gpd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
import time
import copy
from collections import Counter, defaultdict

# ==================================================================================================
# DARP Simulation Engine (darp_simulation_engine.py)
# DARP Simulation with Adaptive Large Neighborhood Search (ALNS)
# Last Updated: 2026-01-27
#
# == Overview ==
# This script performs large-scale simulations of Dial-a-Ride Problem (DARP) variants.
# The main objective is to analyze the efficiency-equity trade-off by adjusting the alpha
# parameter, which controls the weight between mean delay and tail delay (top N%) in the
# objective function, as well as varying the demand levels on the system.
#
# == Objective Function ==
# OFV = (1 - alpha) * Mean_Delay + alpha * Tail_Delay (CVaR at 30%)
# - alpha = 0.0: Mean delay only → Efficiency-focused
# - alpha = 1.0: Tail delay only (CVaR 30%) → Equity-focused
# - 0.0 < alpha < 1.0: Balance between efficiency and equity
#
# == Key Features ==
# 1. Adaptive Large Neighborhood Search with multiple destroy/repair operators
# 2. Delta evaluation for efficient objective function computation
# 3. Parallel processing using ProcessPoolExecutor
# 4. Comprehensive logging of simulation results
#
# ==================================================================================================

# =============================================
# 1) Simulation Configuration
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "Alpha_Demand_Sweep_Results")
SHP_PATH = os.path.join(BASE_DIR, "data", "daejeon.shp")

START_ALPHA = 0.0
END_ALPHA = 1.0
STEP_ALPHA = 0.1

START_DEMAND_MULTIPLIER = 0.5
END_DEMAND_MULTIPLIER = 1.5
STEP_DEMAND_MULTIPLIER = 0.25

SERVICE_TIME = 0.25
VEHICLE_SPEED = 0.5
REOPT_INTERVAL = 1
TAIL_PCT = 0.3

SA_INITIAL_TEMP = 100.0
SA_COOLING_RATE = 0.95
SA_MIN_TEMP = 0.1
SA_MAX_ITERATIONS = 200

ALNS_MAX_ITERATIONS = 200
ALNS_SEGMENT_LENGTH = 50
ALNS_REACTION_FACTOR = 0.1
ALNS_SCORES = {
    'new_best': 33,
    'accepted': 9,
    'feasible': 3
}
ALNS_REMOVAL_RATE_MIN = 0.1
ALNS_REMOVAL_RATE_MAX = 0.4
ALNS_MAX_INSERT_VEHICLES = 10
ALNS_WORST_SAMPLE_RATE = 0.6
ALNS_TIME_BUDGET_MS = 1000
ALNS_LIMIT_INSERT_POSITIONS = 100
ALNS_LOCAL_IMPROVE_ITERS = 8
ALNS_LOCAL_IMPROVE_TIME_MS = 5
ALNS_INSERT_GLOBAL_TOP_K = 3

def _evaluate_candidate_objective(current_solution, candidate_solution, affected_indices, delay_counter,
                                  scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times):
    """Evaluate global objective function for candidate solution using delta evaluation."""
    old_delays_affected = []
    for idx in affected_indices:
        old_delays_affected.extend(get_delays_from_route(current_solution[idx], scenario_tag, current_time, ideal_mode, passenger_travel_times))
    new_delays_affected = []
    for idx in affected_indices:
        new_delays_affected.extend(get_delays_from_route(candidate_solution[idx], scenario_tag, current_time, ideal_mode, passenger_travel_times))
    temp_delay_counter = delay_counter.copy()
    temp_delay_counter.subtract(Counter(old_delays_affected))
    temp_delay_counter.update(Counter(new_delays_affected))
    temp_all_delays = list(temp_delay_counter.elements())
    new_obj = calculate_objective(temp_all_delays, alpha, scenario_tag)
    return new_obj, temp_delay_counter

def _local_intensification_on_indices(solution, indices, capacity, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times):
    """Apply local improvement (intra-vehicle relocate) on affected vehicles."""
    improved = copy.deepcopy(solution)
    start_ms = time.perf_counter_ns() // 1_000_000
    for v_idx in indices:
        v = improved[v_idx]
        route = v['route']
        reqs = get_unserved_requests(route)
        if not reqs:
            continue
        best_route = None
        best_gain = 0.0
        for _ in range(ALNS_LOCAL_IMPROVE_ITERS):
            if (time.perf_counter_ns() // 1_000_000) - start_ms > ALNS_LOCAL_IMPROVE_TIME_MS:
                break
            pid = random.choice(reqs)
            pu_idx = next((i for i, s in enumerate(route) if s['passenger_num']==pid and s['type']=='pickup'), None)
            do_idx = next((i for i, s in enumerate(route) if s['passenger_num']==pid and s['type']=='dropoff'), None)
            if pu_idx is None or do_idx is None:
                continue
            pu, do = route[pu_idx], route[do_idx]
            base_cost = calculate_local_objective_for_route(route, v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
            temp = [s for i, s in enumerate(route) if i not in (pu_idx, do_idx)]
            evals = 0
            for i in range(len(temp)+1):
                for j in range(i+1, len(temp)+2):
                    cand = temp[:i] + [pu] + temp[i:j-1] + [do] + temp[j-1:]
                    if not is_route_capacity_feasible(cand, capacity):
                        continue
                    new_cost = calculate_local_objective_for_route(cand, v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
                    gain = base_cost - new_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_route = cand
                    evals += 1
                    if evals >= ALNS_LIMIT_INSERT_POSITIONS:
                        break
                if evals >= ALNS_LIMIT_INSERT_POSITIONS:
                    break
        if best_route is not None:
            improved[v_idx]['route'] = best_route
    return improved

# =============================================
# 2) Global Variables and Initialization
# =============================================
GRID_SIZE_X, GRID_SIZE_Y = 0, 0

def load_population_weights():
    global GRID_SIZE_X, GRID_SIZE_Y
    gdf = gpd.read_file(SHP_PATH)
    gdf["col"] = ((gdf.geometry.centroid.x - gdf.total_bounds[0]) // 1000).astype(int)
    gdf["row"] = ((gdf.geometry.centroid.y - gdf.total_bounds[1]) // 1000).astype(int)
    cell_weights = {
        (int(row["row"]), int(row["col"])): float(row["val"])
        for _, row in gdf.iterrows() if row["val"] > 0
    }
    GRID_SIZE_X, GRID_SIZE_Y = gdf["col"].max() + 1, gdf["row"].max() + 1
    cells = list(cell_weights.keys())
    weights = np.array(list(cell_weights.values()))
    weights /= weights.sum()
    print(f"Grid size: {GRID_SIZE_X}x{GRID_SIZE_Y}, Valid cells: {len(cells)}")
    return cells, weights

CELLS, POP_WEIGHTS = load_population_weights()
rng = np.random.default_rng()

def sample_cell():
    idx = rng.choice(len(CELLS), p=POP_WEIGHTS)
    return CELLS[idx]

def get_parameters():
    """Get basic simulation parameters from user."""
    print("\n=============================================")
    print("Please enter simulation parameters (press Enter to use defaults)")
    print("---------------------------------------------")
    sim_time = int(input("Simulation time (minutes) [default: 120]: ") or "120")
    base_calls_per_min = float(input("Calls per minute (base value) [default: 1.0]: ") or "1.0")
    num_vehicles = int(input("Number of vehicles [default: 10]: ") or "10")
    vehicle_capacity = int(input("Vehicle capacity [default: 10]: ") or "10")
    num_seeds = int(input("Number of seeds for cross-validation [default: 10]: ") or "10")
    ideal_mode = int(input("Ideal service time mode (1: basic, 2: with vehicle travel) [default: 2]: ") or "2")
    max_workers = int(input(f"Max CPU cores to use (available: {os.cpu_count()}) [default: 12]: ") or "12")
    print("=============================================\n")
    return sim_time, base_calls_per_min, num_vehicles, vehicle_capacity, num_seeds, ideal_mode, max_workers

# =============================================
# 3) Core Utility Functions
# =============================================
def calculate_gini(data):
    if len(data) == 0: return 0
    data = np.sort(np.asarray(data))
    n = len(data)
    if n == 0 or np.sum(data) == 0: return 0
    index = np.arange(1, n + 1)
    return ((2 * np.sum(index * data)) / (n * np.sum(data))) - (n + 1) / n

def is_route_capacity_feasible(route, capacity):
    """Check if the route satisfies vehicle capacity constraints."""
    pickup_passengers = {s['passenger_num'] for s in route if s['type'] == 'pickup'}
    initial_passengers = 0
    for stop in route:
        if stop['type'] == 'dropoff' and stop['passenger_num'] not in pickup_passengers:
            initial_passengers += 1
    
    current_passengers = initial_passengers
    if current_passengers > capacity:
        return False

    for stop in route:
        if stop['type'] == 'pickup':
            current_passengers += 1
            if current_passengers > capacity:
                return False
        else:
            current_passengers -= 1
    return True

def manhattan_distance(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def travel_time(a, b): return manhattan_distance(a, b) / VEHICLE_SPEED

def simulate_dynamic_route(route, start_time, start_loc):
    current_t, current_loc, result, pickup_times = start_time, start_loc, [], {}
    for stop in route:
        arrival = current_t + travel_time(current_loc, stop['loc'])
        if stop['type']=='pickup':
            service_start = max(arrival, stop['call_min'])
            finish_t = service_start + SERVICE_TIME
            pickup_times[stop['passenger_num']] = finish_t
            result.append({**stop, 'arrival_time': arrival, 'service_start': service_start, 'finish_time': finish_t, 'pickup_time': finish_t})
            current_t, current_loc = finish_t, stop['loc']
        else:
            finish_t = arrival + SERVICE_TIME
            result.append({**stop, 'arrival_time': arrival, 'service_start': arrival, 'finish_time': finish_t, 'TST': finish_t - stop['call_min'], 'pickup_time': pickup_times.get(stop['passenger_num'], stop['call_min'])})
            current_t, current_loc = finish_t, stop['loc']
    return result

def get_delays_from_route(vehicle, scenario_tag, current_time, ideal_mode, passenger_travel_times):
    """Simulate the given vehicle route and return the list of delays."""
    sim_route_result = simulate_dynamic_route(vehicle['route'], current_time, vehicle['loc'])
    
    delays = []
    if not sim_route_result:
        return delays
        
    for n in sim_route_result:
        if n['type'] == 'dropoff':
            if ideal_mode == 1:
                delay = n['finish_time'] - (n['call_min'] + SERVICE_TIME + n['ideal_time'])
            else:
                travel_to_pickup = passenger_travel_times.get(n['passenger_num'], {}).get(scenario_tag, 0)
                delay = n['finish_time'] - (n['call_min'] + travel_to_pickup + SERVICE_TIME + n['ideal_time'])
            delays.append(delay)
    return delays

def calculate_objective(delays, alpha, scenario_tag):
    """Calculate objective function value from the list of delays.
    
    Objective: (1 - alpha) * mean_delay + alpha * tail_delay
    - alpha = 0.0: Mean delay only (efficiency-focused)
    - alpha = 1.0: Tail delay (CVaR 30%) only (equity-focused)
    """
    if not delays:
        return 0.0

    if scenario_tag in ('s1', 'scenario1'):
        return np.mean(delays)
    
    mean_delay = np.mean(delays)
    tail_n = max(1, int(len(delays) * TAIL_PCT))
    if tail_n >= len(delays):
        tail_delay = mean_delay
    else:
        arr = np.asarray(delays)
        part = np.partition(arr, len(arr) - tail_n)
        tail_delay = float(np.mean(part[-tail_n:]))
    return (1 - alpha) * mean_delay + alpha * tail_delay

def calculate_local_objective_for_route(route, vehicle, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times):
    """Calculate objective function value for a single modified route."""
    temp_vehicle = vehicle.copy()
    temp_vehicle['route'] = route
    delays = get_delays_from_route(temp_vehicle, scenario_tag, current_time, ideal_mode, passenger_travel_times)
    return calculate_objective(delays, alpha, scenario_tag)

def compute_metrics(vehicles, scenario_tag, ideal_mode, tail_pct, passenger_travel_times):
    """Compute OFV (cumulative sum) and Tail OFV for the given vehicle list."""
    drops = [n for v in vehicles for n in v['completed'] if n['type'] == 'dropoff']
    if not drops: return 0.0, 0.0
    delay_col_name = 'Delay' if ideal_mode == 1 else 'Delay2'
    
    all_completed_nodes = [node for v in vehicles for node in v['completed']]
    dlys = [res[delay_col_name] for n in drops if (res := extract_result_fields(n, all_completed_nodes, scenario_tag, passenger_travel_times))]
    
    if not dlys: return 0.0, 0.0
    
    total_delay = np.sum(dlys)
    tail_n = max(1, int(len(dlys) * tail_pct))
    tail_total_delay = np.sum(np.sort(dlys)[-tail_n:])
    
    return total_delay, tail_total_delay

def extract_result_fields(nd, completed_nodes, scenario_tag, passenger_travel_times):
    pickup_node = next((n for n in completed_nodes if n['type']=='pickup' and n['passenger_num']==nd['passenger_num']), None)
    if pickup_node is None: return None
    
    call_time, dropoff = nd['call_min'], nd['arrival_time'] + SERVICE_TIME
    pickup_time = pickup_node['finish_time']
    pure_travel_time = nd['ideal_time']
    
    delay = dropoff - (call_time + SERVICE_TIME + pure_travel_time)
    travel_to_pickup = passenger_travel_times.get(nd['passenger_num'], {}).get(scenario_tag, 0)
    delay2 = dropoff - (call_time + travel_to_pickup + SERVICE_TIME + pure_travel_time)
    
    in_vehicle_time = dropoff - pickup_time
    detour_factor = in_vehicle_time / pure_travel_time if pure_travel_time > 0 else 1.0

    return {"PassengerNumber": nd['passenger_num'], "CallTime": call_time, "PickupTime": pickup_time, 
            "DropoffTime": dropoff, "TST": nd['TST'], "Delay": delay, "Delay2": delay2,
            "WaitingTime": pickup_time - call_time,
            "InVehicleTime": in_vehicle_time, "DetourFactor": detour_factor}

def update_executed_route(sim_route, target_time, start_t, start_loc):
    current_t, current_loc, executed, remaining = start_t, start_loc, [], []
    distance_traveled = 0.0
    temp_loc = start_loc

    for idx, node in enumerate(sim_route):
        travel_dist_segment = manhattan_distance(temp_loc, node['loc'])
        
        if node['finish_time'] <= target_time:
            executed.append(node)
            current_t = node['finish_time']
            temp_loc = node['loc']
            distance_traveled += travel_dist_segment
        else:
            if node['finish_time'] > current_t:
                frac = (target_time - current_t) / (node['finish_time'] - current_t) if (node['finish_time'] - current_t) > 0 else 0
                final_loc = tuple(np.array(temp_loc) + frac * (np.array(node['loc']) - np.array(temp_loc)))
                distance_traveled += manhattan_distance(temp_loc, final_loc)
                current_loc = final_loc
            else:
                current_loc = temp_loc
            
            remaining = sim_route[idx:]
            return executed, remaining, current_loc, distance_traveled

    current_loc = temp_loc
    return executed, remaining, current_loc, distance_traveled

def _insert_request_into_route(route, pu_stop, do_stop, vehicle, current_time,
                               capacity, scenario_tag, alpha, ideal_mode, passenger_travel_times):
    """Find the best insertion position for pickup/dropoff pair in the given route."""
    best_route = None
    min_local_obj = float('inf')

    max_eval = ALNS_LIMIT_INSERT_POSITIONS
    eval_count = 0
    for i in range(len(route) + 1):
        for j in range(i + 1, len(route) + 2):
            cand_route = route[:i] + [pu_stop] + route[i:j-1] + [do_stop] + route[j-1:]
            
            if not is_route_capacity_feasible(cand_route, capacity):
                continue
            
            local_obj = calculate_local_objective_for_route(
                cand_route, vehicle, scenario_tag, current_time,
                alpha, ideal_mode, passenger_travel_times
            )

            if local_obj < min_local_obj:
                min_local_obj, best_route = local_obj, cand_route
            eval_count += 1
            if eval_count >= max_eval:
                return best_route
                
    return best_route

# =============================================
# 4) Global Optimization Engine (ALNS with SA)
# =============================================

def get_unserved_requests(route):
    """Return list of unserved requests (pickup, dropoff pairs) in the route."""
    return list({stop['passenger_num'] for stop in route if stop['type'] == 'pickup'})

def calculate_global_objective(vehicles, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times):
    """Calculate the objective function value for the entire system."""
    all_delays = []
    for v in vehicles:
        sim_route_result = simulate_dynamic_route(v['route'], current_time, v['loc'])
        for n in sim_route_result:
            if n['type'] == 'dropoff':
                if ideal_mode == 1:
                    delay = n['finish_time'] - (n['call_min'] + SERVICE_TIME + n['ideal_time'])
                else:
                    travel_to_pickup = passenger_travel_times.get(n['passenger_num'], {}).get(scenario_tag, 0)
                    delay = n['finish_time'] - (n['call_min'] + travel_to_pickup + SERVICE_TIME + n['ideal_time'])
                all_delays.append(delay)
    
    if not all_delays:
        return 0.0
    
    if scenario_tag in ('s1', 'scenario1'):
        return np.mean(all_delays)
    else:
        mean_delay = np.mean(all_delays)
        tail_n = max(1, int(len(all_delays) * TAIL_PCT))
        if tail_n >= len(all_delays):
            tail_delay = mean_delay
        else:
            arr = np.asarray(all_delays)
            part = np.partition(arr, len(arr) - tail_n)
            tail_delay = float(np.mean(part[-tail_n:]))
        return (1 - alpha) * mean_delay + alpha * tail_delay

def intra_vehicle_relocate(vehicles, capacity, **kwargs):
    """Intra-vehicle request relocation operator."""
    v_idx = random.choice(range(len(vehicles)))
    v = vehicles[v_idx]
    
    if len(v['route']) < 3: return None, None
    
    unserved_reqs = get_unserved_requests(v['route'])
    if not unserved_reqs: return None, None
    req_num_to_move = random.choice(unserved_reqs)
    
    pu_idx = next((i for i, s in enumerate(v['route']) if s['passenger_num'] == req_num_to_move and s['type'] == 'pickup'), -1)
    do_idx = next((i for i, s in enumerate(v['route']) if s['passenger_num'] == req_num_to_move and s['type'] == 'dropoff'), -1)

    if pu_idx == -1 or do_idx == -1: return None, None

    pu_stop, do_stop = v['route'][pu_idx], v['route'][do_idx]
    temp_route = [s for i, s in enumerate(v['route']) if i not in (pu_idx, do_idx)]
    
    best_new_route = _insert_request_into_route(
        temp_route, pu_stop, do_stop, v, kwargs['current_time'],
        capacity, kwargs['scenario_tag'], kwargs['alpha'], kwargs['ideal_mode'], kwargs['passenger_travel_times']
    )

    if best_new_route:
        new_vehicles = copy.deepcopy(vehicles)
        new_vehicles[v_idx]['route'] = best_new_route
        return new_vehicles, [v_idx]
    return None, None

def intra_vehicle_2opt(vehicles, capacity, **kwargs):
    """Intra-vehicle 2-opt exchange operator."""
    v_idx = random.choice(range(len(vehicles)))
    v = vehicles[v_idx]
    if len(v['route']) < 2: return None, None

    best_route = None
    best_gain = 0.0
    start_ms = time.perf_counter_ns() // 1_000_000
    for _ in range(ALNS_LOCAL_IMPROVE_ITERS):
        if (time.perf_counter_ns() // 1_000_000) - start_ms > ALNS_LOCAL_IMPROVE_TIME_MS:
            break
        new_route = v['route'][:]
        i, j = sorted(random.sample(range(len(new_route)), 2))
        new_route[i], new_route[j] = new_route[j], new_route[i]
        if not is_route_capacity_feasible(new_route, capacity):
            continue
        seen = {}
        valid = True
        for idx, node in enumerate(new_route):
            pid = node['passenger_num']
            if node['type'] == 'pickup':
                seen[pid] = idx
            elif pid not in seen or seen[pid] > idx:
                valid = False
                break
        if not valid:
            continue
        old_cost = calculate_local_objective_for_route(v['route'], v, kwargs['scenario_tag'], kwargs['current_time'], kwargs['alpha'], kwargs['ideal_mode'], kwargs['passenger_travel_times'])
        new_cost = calculate_local_objective_for_route(new_route, v, kwargs['scenario_tag'], kwargs['current_time'], kwargs['alpha'], kwargs['ideal_mode'], kwargs['passenger_travel_times'])
        gain = old_cost - new_cost
        if gain > best_gain:
            best_gain = gain
            best_route = new_route

    if best_route is not None:
        new_vehicles = copy.deepcopy(vehicles)
        new_vehicles[v_idx]['route'] = best_route
        return new_vehicles, [v_idx]
    return None, None

def intra_vehicle_or_opt(vehicles, capacity, **kwargs):
    """Intra-vehicle Or-opt (segment relocation) operator."""
    v_idx = random.choice(range(len(vehicles)))
    v = vehicles[v_idx]
    if len(v['route']) < 3: return None, None

    best_route = None
    best_gain = 0.0
    start_ms = time.perf_counter_ns() // 1_000_000
    for _ in range(ALNS_LOCAL_IMPROVE_ITERS):
        if (time.perf_counter_ns() // 1_000_000) - start_ms > ALNS_LOCAL_IMPROVE_TIME_MS:
            break
        seq_len = random.randint(2, min(3, len(v['route']) - 1))
        if len(v['route']) <= seq_len:
            continue
        start_idx = random.randint(0, len(v['route']) - seq_len)
        sequence = v['route'][start_idx : start_idx + seq_len]
        remaining_route = v['route'][:start_idx] + v['route'][start_idx + seq_len:]
        insert_pos = random.randint(0, len(remaining_route))
        new_route = remaining_route[:insert_pos] + sequence + remaining_route[insert_pos:]
        if not is_route_capacity_feasible(new_route, capacity):
            continue
        seen = {}
        valid = True
        for idx, node in enumerate(new_route):
            pid = node['passenger_num']
            if node['type'] == 'pickup':
                seen[pid] = idx
            elif pid not in seen or seen[pid] > idx:
                valid = False
                break
        if not valid:
            continue
        old_cost = calculate_local_objective_for_route(v['route'], v, kwargs['scenario_tag'], kwargs['current_time'], kwargs['alpha'], kwargs['ideal_mode'], kwargs['passenger_travel_times'])
        new_cost = calculate_local_objective_for_route(new_route, v, kwargs['scenario_tag'], kwargs['current_time'], kwargs['alpha'], kwargs['ideal_mode'], kwargs['passenger_travel_times'])
        gain = old_cost - new_cost
        if gain > best_gain:
            best_gain = gain
            best_route = new_route
    if best_route is not None:
        new_vehicles = copy.deepcopy(vehicles)
        new_vehicles[v_idx]['route'] = best_route
        return new_vehicles, [v_idx]
    return None, None

def inter_vehicle_relocate(vehicles, capacity, **kwargs):
    """Inter-vehicle request relocation operator."""
    if len(vehicles) < 2: return None, None
    
    v_source_idx, v_dest_idx = random.sample(range(len(vehicles)), 2)
    v_source, v_dest = vehicles[v_source_idx], vehicles[v_dest_idx]
    
    unserved_reqs = get_unserved_requests(v_source['route'])
    if not unserved_reqs: return None, None

    p_num_to_move = random.choice(unserved_reqs)
    
    pu = next((s for s in v_source['route'] if s['passenger_num'] == p_num_to_move and s['type'] == 'pickup'), None)
    do = next((s for s in v_source['route'] if s['passenger_num'] == p_num_to_move and s['type'] == 'dropoff'), None)

    if pu is None or do is None: return None, None
    
    new_route_source = [s for s in v_source['route'] if s['passenger_num'] != p_num_to_move]
    
    new_route_dest = _insert_request_into_route(
        v_dest['route'], pu, do, v_dest, kwargs['current_time'],
        capacity, kwargs['scenario_tag'], kwargs['alpha'], kwargs['ideal_mode'], kwargs['passenger_travel_times']
    )

    if new_route_dest:
        new_vehicles = copy.deepcopy(vehicles)
        new_vehicles[v_source_idx]['route'] = new_route_source
        new_vehicles[v_dest_idx]['route'] = new_route_dest
        return new_vehicles, [v_source_idx, v_dest_idx]
    return None, None

def inter_vehicle_swap(vehicles, capacity, **kwargs):
    """Inter-vehicle request swap operator."""
    if len(vehicles) < 2: return None, None

    v_source_idx, v_dest_idx = random.sample(range(len(vehicles)), 2)
    v_source, v_dest = vehicles[v_source_idx], vehicles[v_dest_idx]

    unserved_src = get_unserved_requests(v_source['route'])
    unserved_dest = get_unserved_requests(v_dest['route'])
    if not unserved_src or not unserved_dest: return None, None

    p_num_src = random.choice(unserved_src)
    p_num_dest = random.choice(unserved_dest)

    pu_src = next((s for s in v_source['route'] if s['passenger_num'] == p_num_src and s['type'] == 'pickup'), None)
    do_src = next((s for s in v_source['route'] if s['passenger_num'] == p_num_src and s['type'] == 'dropoff'), None)
    pu_dest = next((s for s in v_dest['route'] if s['passenger_num'] == p_num_dest and s['type'] == 'pickup'), None)
    do_dest = next((s for s in v_dest['route'] if s['passenger_num'] == p_num_dest and s['type'] == 'dropoff'), None)

    if not all([pu_src, do_src, pu_dest, do_dest]): return None, None

    temp_route_src = [s for s in v_source['route'] if s['passenger_num'] != p_num_src]
    temp_route_dest = [s for s in v_dest['route'] if s['passenger_num'] != p_num_dest]

    final_route_src = _insert_request_into_route(
        temp_route_src, pu_dest, do_dest, v_source, kwargs['current_time'],
        capacity, kwargs['scenario_tag'], kwargs['alpha'], kwargs['ideal_mode'], kwargs['passenger_travel_times']
    )
    final_route_dest = _insert_request_into_route(
        temp_route_dest, pu_src, do_src, v_dest, kwargs['current_time'],
        capacity, kwargs['scenario_tag'], kwargs['alpha'], kwargs['ideal_mode'], kwargs['passenger_travel_times']
    )

    if final_route_src and final_route_dest:
        new_vehicles = copy.deepcopy(vehicles)
        new_vehicles[v_source_idx]['route'] = final_route_src
        new_vehicles[v_dest_idx]['route'] = final_route_dest
        return new_vehicles, [v_source_idx, v_dest_idx]
    return None, None

def run_sa_optimization(vehicles, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times, capacity):
    """SA optimization with delta evaluation for the entire vehicle system."""
    initial_temp, cooling_rate, min_temp, max_iterations = SA_INITIAL_TEMP, SA_COOLING_RATE, SA_MIN_TEMP, SA_MAX_ITERATIONS

    current_solution = copy.deepcopy(vehicles)
    
    all_delays = []
    for v in current_solution:
        all_delays.extend(get_delays_from_route(v, scenario_tag, current_time, ideal_mode, passenger_travel_times))
    
    if not all_delays:
        return current_solution
        
    current_obj = calculate_objective(all_delays, alpha, scenario_tag)
    
    best_solution, best_obj = copy.deepcopy(current_solution), current_obj
    delay_counter = Counter(all_delays)

    intra_operators = [intra_vehicle_relocate, intra_vehicle_2opt, intra_vehicle_or_opt]
    inter_operators = [inter_vehicle_relocate, inter_vehicle_swap]
    kwargs = {
        'capacity': capacity, 'current_time': current_time, 'scenario_tag': scenario_tag,
        'alpha': alpha, 'ideal_mode': ideal_mode, 'passenger_travel_times': passenger_travel_times
    }

    temp = initial_temp
    for _ in range(max_iterations):
        if temp < min_temp: break
        
        prob_inter = 0.6 * (temp / initial_temp)
        operator = random.choice(inter_operators) if random.random() < prob_inter else random.choice(intra_operators)
            
        candidate_solution, affected_indices = operator(current_solution, **kwargs)
            
        if candidate_solution is None:
            temp *= cooling_rate
            continue

        old_delays_affected = []
        for idx in affected_indices:
            old_delays_affected.extend(get_delays_from_route(current_solution[idx], scenario_tag, current_time, ideal_mode, passenger_travel_times))

        new_delays_affected = []
        for idx in affected_indices:
            new_delays_affected.extend(get_delays_from_route(candidate_solution[idx], scenario_tag, current_time, ideal_mode, passenger_travel_times))

        temp_delay_counter = delay_counter.copy()
        temp_delay_counter.subtract(Counter(old_delays_affected))
        temp_delay_counter.update(Counter(new_delays_affected))
        temp_all_delays = list(temp_delay_counter.elements())

        new_obj = calculate_objective(temp_all_delays, alpha, scenario_tag)
        
        if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / temp):
            current_solution = candidate_solution
            current_obj = new_obj
            delay_counter = temp_delay_counter
            
            if current_obj < best_obj:
                best_solution = copy.deepcopy(current_solution)
                best_obj = current_obj
                
        temp *= cooling_rate
        
    return best_solution

def assign_and_optimize_SA(vehicles, req, scenario_tag, current_time, alpha=0.5, passenger_travel_times={}, capacity=float('inf'), ideal_mode=2):
    """SA-based optimization function with delta evaluation."""
    
    best_v_idx, min_local_obj, best_initial_route = -1, float('inf'), None
    
    pu = {'loc': req['pickup'], 'type': 'pickup', 'call_min': req['call_min'], 'ideal_time': req['ideal_time'], 'passenger_num': req['passenger_num']}
    do = {'loc': req['dropoff'], 'type': 'dropoff', 'call_min': req['call_min'], 'ideal_time': req['ideal_time'], 'passenger_num': req['passenger_num']}
        
    vehicle_indices = list(range(len(vehicles)))
    vehicle_indices.sort(key=lambda i: manhattan_distance(vehicles[i]['loc'], pu['loc']))
    candidate_indices = vehicle_indices[:min(ALNS_MAX_INSERT_VEHICLES, len(vehicle_indices))]

    for i in candidate_indices:
        v = vehicles[i]
        cand_route = _insert_request_into_route(
            v['route'], pu, do, v, current_time,
            capacity, scenario_tag, alpha, ideal_mode, passenger_travel_times
        )
        
        if cand_route:
            local_obj = calculate_local_objective_for_route(
                cand_route, v, scenario_tag, current_time, 
                alpha, ideal_mode, passenger_travel_times
            )
            if local_obj < min_local_obj:
                min_local_obj = local_obj
                best_v_idx = i
                best_initial_route = cand_route

    initial_solution = copy.deepcopy(vehicles)
    if best_v_idx != -1:
        initial_solution[best_v_idx]['route'] = best_initial_route
    else:
        return vehicles

    optimized_solution = run_sa_optimization(initial_solution, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times, capacity)
    
    return optimized_solution

# =============================================
# 4-1) ALNS Destroy and Repair Operators
# =============================================

def _get_all_unserved_request_ids(vehicles):
    return list({stop['passenger_num'] for v in vehicles for stop in v['route'] if stop['type'] == 'pickup'})

def _build_request_index(vehicles):
    index = {}
    for v_idx, v in enumerate(vehicles):
        for i, s in enumerate(v['route']):
            pid = s['passenger_num']
            if pid not in index:
                index[pid] = {'vehicle_idx': v_idx, 'pickup_idx': None, 'dropoff_idx': None, 'pickup': None, 'dropoff': None}
            if s['type'] == 'pickup':
                index[pid]['pickup_idx'] = i
                index[pid]['pickup'] = s
            else:
                index[pid]['dropoff_idx'] = i
                index[pid]['dropoff'] = s
    return index

def _remove_requests(vehicles, request_ids):
    """Safely remove requests by re-searching indices in current routes."""
    new_vehicles = copy.deepcopy(vehicles)
    removed_map = {}
    affected = set()
    for pid in request_ids:
        target_v_idx = None
        pu_idx = do_idx = None
        pu_stop = do_stop = None
        for v_idx, v in enumerate(new_vehicles):
            route = v['route']
            pu_idx = next((i for i, s in enumerate(route) if s['passenger_num'] == pid and s['type'] == 'pickup'), None)
            do_idx = next((i for i, s in enumerate(route) if s['passenger_num'] == pid and s['type'] == 'dropoff'), None)
            if pu_idx is not None or do_idx is not None:
                target_v_idx = v_idx
                if pu_idx is not None:
                    pu_stop = route[pu_idx]
                if do_idx is not None:
                    do_stop = route[do_idx]
                break
        if target_v_idx is None:
            continue
        route = new_vehicles[target_v_idx]['route']
        indices = [x for x in [pu_idx, do_idx] if x is not None]
        if not indices:
            continue
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(route):
                del route[idx]
        removed_map[pid] = (pu_stop, do_stop)
        affected.add(target_v_idx)
    return new_vehicles, removed_map, sorted(list(affected))

def destroy_random(vehicles, num_remove, **kwargs):
    """Random removal destroy operator."""
    all_ids = _get_all_unserved_request_ids(vehicles)
    if not all_ids:
        return None, None, None
    num_remove = max(1, min(num_remove, len(all_ids)))
    selected = random.sample(all_ids, num_remove)
    new_sol, removed_map, affected = _remove_requests(vehicles, selected)
    return new_sol, removed_map, affected

def _request_features_from_stop_pair(pu, do):
    return {
        'pu_loc': pu['loc'],
        'do_loc': do['loc'],
        'call_min': pu['call_min'],
        'ideal_time': pu['ideal_time']
    }

def destroy_shaw(vehicles, num_remove, **kwargs):
    """Shaw removal (similarity-based) destroy operator."""
    all_ids = _get_all_unserved_request_ids(vehicles)
    if not all_ids:
        return None, None, None
    req_index = _build_request_index(vehicles)
    features = {}
    for pid in all_ids:
        info = req_index.get(pid)
        if info and info['pickup'] and info['dropoff']:
            features[pid] = _request_features_from_stop_pair(info['pickup'], info['dropoff'])
    if not features:
        return None, None, None
    w_pu, w_do, w_t, w_len = 1.0, 1.0, 0.5, 0.5
    seed = random.choice(list(features.keys()))
    selected = [seed]
    while len(selected) < min(max(1, num_remove), len(features)):
        best_pid, best_r = None, float('inf')
        for pid, f in features.items():
            if pid in selected:
                continue
            r = 0.0
            for s in selected:
                fs = features[s]
                r += (
                    w_pu * manhattan_distance(f['pu_loc'], fs['pu_loc']) +
                    w_do * manhattan_distance(f['do_loc'], fs['do_loc']) +
                    w_t * abs(f['call_min'] - fs['call_min']) +
                    w_len * abs(f['ideal_time'] - fs['ideal_time'])
                )
            if r < best_r:
                best_r, best_pid = r, pid
        if best_pid is None:
            break
        selected.append(best_pid)
    new_sol, removed_map, affected = _remove_requests(vehicles, selected)
    return new_sol, removed_map, affected

def destroy_worst(vehicles, num_remove, **kwargs):
    """Worst removal (cost-based) destroy operator."""
    scenario_tag = kwargs['scenario_tag']
    current_time = kwargs['current_time']
    alpha = kwargs['alpha']
    ideal_mode = kwargs['ideal_mode']
    passenger_travel_times = kwargs['passenger_travel_times']
    capacity = kwargs['capacity']

    req_index = _build_request_index(vehicles)
    all_pids = list(req_index.keys())
    if 0.0 < ALNS_WORST_SAMPLE_RATE < 1.0 and len(all_pids) > 0:
        sample_size = max(1, int(len(all_pids) * ALNS_WORST_SAMPLE_RATE))
        sampled_pids = set(random.sample(all_pids, sample_size))
    else:
        sampled_pids = set(all_pids)

    candidates = []
    for pid, info in req_index.items():
        if pid not in sampled_pids:
            continue
        v_idx = info['vehicle_idx']
        if v_idx is None or info['pickup_idx'] is None or info['dropoff_idx'] is None:
            continue
        v = vehicles[v_idx]
        route = v['route']
        current_cost = calculate_local_objective_for_route(route, v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
        new_route = [s for s in route if s['passenger_num'] != pid]
        if not is_route_capacity_feasible(new_route, capacity):
            continue
        new_cost = calculate_local_objective_for_route(new_route, v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
        savings = current_cost - new_cost
        candidates.append((pid, savings))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [pid for pid, _ in candidates[:max(1, min(num_remove, len(candidates)))] ]
    new_sol, removed_map, affected = _remove_requests(vehicles, selected)
    return new_sol, removed_map, affected

def destroy_route(vehicles, num_remove, **kwargs):
    """Route removal destroy operator."""
    counts = [(i, len(get_unserved_requests(v['route']))) for i, v in enumerate(vehicles)]
    if not counts:
        return None, None, None
    counts.sort(key=lambda x: x[1], reverse=True)
    v_idx, cnt = counts[0]
    if cnt == 0:
        return None, None, None
    take = max(1, min(num_remove, cnt))
    reqs = get_unserved_requests(vehicles[v_idx]['route'])
    selected = random.sample(reqs, take) if take < len(reqs) else reqs
    new_vehicles = copy.deepcopy(vehicles)
    removed_map = {}
    route = new_vehicles[v_idx]['route']
    for pid in selected:
        pu_idx = next((i for i, s in enumerate(route) if s['passenger_num']==pid and s['type']=='pickup'), None)
        do_idx = next((i for i, s in enumerate(route) if s['passenger_num']==pid and s['type']=='dropoff'), None)
        indices = [i for i in [pu_idx, do_idx] if i is not None]
        for idx in sorted(indices, reverse=True):
            del route[idx]
        orig_route = next(v['route'] for v in vehicles if v['id'] == new_vehicles[v_idx]['id'])
        pu = next(s for s in orig_route if s['passenger_num']==pid and s['type']=='pickup')
        do = next(s for s in orig_route if s['passenger_num']==pid and s['type']=='dropoff')
        removed_map[pid] = (pu, do)
    return new_vehicles, removed_map, [v_idx]

def repair_greedy(vehicles, removed_map, **kwargs):
    """Greedy insertion repair operator."""
    current_time = kwargs['current_time']
    alpha = kwargs['alpha']
    ideal_mode = kwargs['ideal_mode']
    passenger_travel_times = kwargs['passenger_travel_times']
    capacity = kwargs['capacity']
    scenario_tag = kwargs['scenario_tag']

    new_sol = copy.deepcopy(vehicles)
    request_ids = list(removed_map.keys())
    random.shuffle(request_ids)
    affected = set()
    for pid in request_ids:
        pu, do = removed_map[pid]
        best = None
        best_i = None
        best_cost = float('inf')
        indices = list(range(len(new_sol)))
        indices.sort(key=lambda i: manhattan_distance(new_sol[i]['loc'], pu['loc']))
        indices = indices[:min(ALNS_MAX_INSERT_VEHICLES, len(indices))]
        for i in indices:
            v = new_sol[i]
            cand = _insert_request_into_route(v['route'], pu, do, v, current_time, capacity, scenario_tag, alpha, ideal_mode, passenger_travel_times)
            if cand:
                old_cost = calculate_local_objective_for_route(v['route'], v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
                new_cost = calculate_local_objective_for_route(cand, v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
                inc = new_cost - old_cost
                if inc < best_cost:
                    best_cost = inc
                    best = cand
                    best_i = i
        if best is None:
            return None, None
        new_sol[best_i]['route'] = best
        affected.add(best_i)
    return new_sol, sorted(list(affected))

def repair_regret(vehicles, removed_map, k=2, **kwargs):
    """Regret-k insertion repair operator."""
    current_time = kwargs['current_time']
    alpha = kwargs['alpha']
    ideal_mode = kwargs['ideal_mode']
    passenger_travel_times = kwargs['passenger_travel_times']
    capacity = kwargs['capacity']
    scenario_tag = kwargs['scenario_tag']

    new_sol = copy.deepcopy(vehicles)
    remaining = set(removed_map.keys())
    affected = set()

    while remaining:
        best_pid, best_choice = None, None
        best_regret = -1.0
        for pid in list(remaining):
            pu, do = removed_map[pid]
            cost_list = []
            route_list = []
            idx_list = []
            indices = list(range(len(new_sol)))
            indices.sort(key=lambda i: manhattan_distance(new_sol[i]['loc'], pu['loc']))
            indices = indices[:min(ALNS_MAX_INSERT_VEHICLES, len(indices))]
            for i in indices:
                v = new_sol[i]
                cand = _insert_request_into_route(v['route'], pu, do, v, current_time, capacity, scenario_tag, alpha, ideal_mode, passenger_travel_times)
                if not cand:
                    continue
                old_cost = calculate_local_objective_for_route(v['route'], v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
                new_cost = calculate_local_objective_for_route(cand, v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
                cost_list.append(new_cost - old_cost)
                route_list.append(cand)
                idx_list.append(i)
            if not cost_list:
                continue
            order = np.argsort(cost_list)
            top_k = [cost_list[j] for j in order[:min(k, len(order))]]
            regret = sum(top_k[j] - top_k[0] for j in range(1, len(top_k)))
            if regret > best_regret:
                best_regret = regret
                best_pid = pid
                best_choice = (idx_list[order[0]], route_list[order[0]])
        if best_pid is None or best_choice is None:
            return None, None
        i_sel, route_sel = best_choice
        new_sol[i_sel]['route'] = route_sel
        affected.add(i_sel)
        remaining.remove(best_pid)
    return new_sol, sorted(list(affected))

def repair_randomized_greedy(vehicles, removed_map, noise=0.2, **kwargs):
    """Randomized greedy insertion repair operator."""
    current_time = kwargs['current_time']
    alpha = kwargs['alpha']
    ideal_mode = kwargs['ideal_mode']
    passenger_travel_times = kwargs['passenger_travel_times']
    capacity = kwargs['capacity']
    scenario_tag = kwargs['scenario_tag']

    new_sol = copy.deepcopy(vehicles)
    request_ids = list(removed_map.keys())
    random.shuffle(request_ids)
    affected = set()
    for pid in request_ids:
        pu, do = removed_map[pid]
        best = None
        best_i = None
        best_cost = float('inf')
        indices = list(range(len(new_sol)))
        indices.sort(key=lambda i: manhattan_distance(new_sol[i]['loc'], pu['loc']))
        indices = indices[:min(ALNS_MAX_INSERT_VEHICLES, len(indices))]
        for i in indices:
            v = new_sol[i]
            cand = _insert_request_into_route(v['route'], pu, do, v, current_time, capacity, scenario_tag, alpha, ideal_mode, passenger_travel_times)
            if cand:
                old_cost = calculate_local_objective_for_route(v['route'], v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
                new_cost = calculate_local_objective_for_route(cand, v, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times)
                inc = new_cost - old_cost
                inc *= (1.0 + random.uniform(0.0, noise))
                if inc < best_cost:
                    best_cost = inc
                    best = cand
                    best_i = i
        if best is None:
            return None, None
        new_sol[best_i]['route'] = best
        affected.add(best_i)
    return new_sol, sorted(list(affected))

def run_alns_optimization(vehicles, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times, capacity):
    """ALNS optimization with adaptive operator selection."""
    current_solution = copy.deepcopy(vehicles)
    all_delays = []
    for v in current_solution:
        all_delays.extend(get_delays_from_route(v, scenario_tag, current_time, ideal_mode, passenger_travel_times))
    if not all_delays:
        return current_solution
    current_obj = calculate_objective(all_delays, alpha, scenario_tag)
    best_solution, best_obj = copy.deepcopy(current_solution), current_obj
    delay_counter = Counter(all_delays)

    destroy_ops = [
        {'name': 'random', 'func': destroy_random},
        {'name': 'shaw', 'func': destroy_shaw},
        {'name': 'worst', 'func': destroy_worst},
        {'name': 'route', 'func': destroy_route}
    ]
    repair_ops = [
        {'name': 'greedy', 'func': repair_greedy},
        {'name': 'regret2', 'func': lambda sol, removed, **kw: repair_regret(sol, removed, k=2, **kw)},
        {'name': 'rand_greedy', 'func': repair_randomized_greedy}
    ]

    d_weights = {op['name']: 1.0 for op in destroy_ops}
    r_weights = {op['name']: 1.0 for op in repair_ops}
    d_scores = {op['name']: 0.0 for op in destroy_ops}
    r_scores = {op['name']: 0.0 for op in repair_ops}
    d_uses = {op['name']: 0 for op in destroy_ops}
    r_uses = {op['name']: 0 for op in repair_ops}

    initial_temp = SA_INITIAL_TEMP
    temp = initial_temp
    cooling_rate = SA_COOLING_RATE
    min_temp = SA_MIN_TEMP

    start_ms = time.perf_counter_ns() // 1_000_000
    for it in range(ALNS_MAX_ITERATIONS):
        if temp < min_temp:
            break
        now_ms = time.perf_counter_ns() // 1_000_000
        if now_ms - start_ms >= ALNS_TIME_BUDGET_MS:
            break

        total_unserved = len(_get_all_unserved_request_ids(current_solution))
        if total_unserved == 0:
            break
        frac = random.uniform(ALNS_REMOVAL_RATE_MIN, ALNS_REMOVAL_RATE_MAX)
        num_remove = max(1, int(frac * total_unserved))

        d_names, d_w = zip(*d_weights.items())
        r_names, r_w = zip(*r_weights.items())
        d_choice = random.choices(d_names, weights=d_w, k=1)[0]
        r_choice = random.choices(r_names, weights=r_w, k=1)[0]
        d_func = next(op['func'] for op in destroy_ops if op['name']==d_choice)
        r_func = next(op['func'] for op in repair_ops if op['name']==r_choice)

        kwargs = {
            'capacity': capacity, 'current_time': current_time, 'scenario_tag': scenario_tag,
            'alpha': alpha, 'ideal_mode': ideal_mode, 'passenger_travel_times': passenger_travel_times
        }
        partial_sol, removed_map, affected_destroy = d_func(current_solution, num_remove, **kwargs)
        if partial_sol is None or not removed_map:
            temp *= cooling_rate
            continue

        repaired_sol, affected_repair = r_func(partial_sol, removed_map, **kwargs)
        if repaired_sol is None:
            repaired_sol, affected_repair = repair_greedy(partial_sol, removed_map, **kwargs)
            if repaired_sol is None:
                temp *= cooling_rate
                continue

        affected_indices = sorted(list(set((affected_destroy or []) + (affected_repair or []))))
        if not affected_indices:
            affected_indices = list(range(len(current_solution)))

        new_obj, temp_delay_counter = _evaluate_candidate_objective(
            current_solution, repaired_sol, affected_indices, delay_counter,
            scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times
        )

        intensified = _local_intensification_on_indices(
            repaired_sol, affected_indices, capacity, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times
        )
        new_obj_int, temp_delay_counter_int = _evaluate_candidate_objective(
            current_solution, intensified, affected_indices, delay_counter,
            scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times
        )
        if new_obj_int < new_obj:
            repaired_sol = intensified
            new_obj = new_obj_int
            temp_delay_counter = temp_delay_counter_int

        accepted = False
        reward_key = None
        if new_obj < current_obj:
            accepted = True
            reward_key = 'accepted'
        elif random.random() < math.exp(-(new_obj - current_obj) / max(temp, 1e-9)):
            accepted = True
            reward_key = 'feasible'

        if accepted:
            current_solution = repaired_sol
            current_obj = new_obj
            delay_counter = temp_delay_counter
            if new_obj < best_obj:
                best_solution = copy.deepcopy(current_solution)
                best_obj = new_obj
                reward_key = 'new_best'
            d_scores[d_choice] += ALNS_SCORES[reward_key]
            r_scores[r_choice] += ALNS_SCORES[reward_key]
            d_uses[d_choice] += 1
            r_uses[r_choice] += 1
        temp *= cooling_rate

        if (it+1) % ALNS_SEGMENT_LENGTH == 0:
            for name in d_weights:
                if d_uses[name] > 0:
                    avg_score = d_scores[name] / d_uses[name]
                    d_weights[name] = (1 - ALNS_REACTION_FACTOR) * d_weights[name] + ALNS_REACTION_FACTOR * avg_score
                d_scores[name] = 0.0
                d_uses[name] = 0
            for name in r_weights:
                if r_uses[name] > 0:
                    avg_score = r_scores[name] / r_uses[name]
                    r_weights[name] = (1 - ALNS_REACTION_FACTOR) * r_weights[name] + ALNS_REACTION_FACTOR * avg_score
                r_scores[name] = 0.0
                r_uses[name] = 0

    return best_solution

def assign_and_optimize_ALNS(vehicles, req, scenario_tag, current_time, alpha=0.5, passenger_travel_times={}, capacity=float('inf'), ideal_mode=2):
    """ALNS-based optimization function."""
    best_v_idx, min_local_obj, best_initial_route = -1, float('inf'), None
    pu = {'loc': req['pickup'], 'type': 'pickup', 'call_min': req['call_min'], 'ideal_time': req['ideal_time'], 'passenger_num': req['passenger_num']}
    do = {'loc': req['dropoff'], 'type': 'dropoff', 'call_min': req['call_min'], 'ideal_time': req['ideal_time'], 'passenger_num': req['passenger_num']}

    for i, v in enumerate(vehicles):
        cand_route = _insert_request_into_route(
            v['route'], pu, do, v, current_time,
            capacity, scenario_tag, alpha, ideal_mode, passenger_travel_times
        )
        if cand_route:
            local_obj = calculate_local_objective_for_route(
                cand_route, v, scenario_tag, current_time,
                alpha, ideal_mode, passenger_travel_times
            )
            if local_obj < min_local_obj:
                min_local_obj = local_obj
                best_v_idx = i
                best_initial_route = cand_route

    initial_solution = copy.deepcopy(vehicles)
    if best_v_idx != -1:
        initial_solution[best_v_idx]['route'] = best_initial_route
    else:
        return vehicles

    optimized_solution = run_alns_optimization(initial_solution, scenario_tag, current_time, alpha, ideal_mode, passenger_travel_times, capacity)
    return optimized_solution
    
def safe_gini(df, col):
    return calculate_gini(df[col].dropna()) if col in df.columns and not df[col].empty else np.nan

def run_simulation_task(params):
    """Run a single simulation task for given (alpha, demand, seed) combination."""
    alpha, demand_multiplier, seed, sim_time, base_calls_per_min, num_vehicles, ideal_mode, vehicle_capacity = params
    
    random.seed(seed)
    np.random.seed(seed)
    
    passenger_counter = 1
    passenger_travel_times = {}

    depot_loc = (GRID_SIZE_X // 2, GRID_SIZE_Y // 2)
    vehicles_s1 = [{'id':i, 'loc':depot_loc, 'route':[], 'completed':[]} for i in range(num_vehicles)]
    vehicles_s2 = [{'id':i, 'loc':depot_loc, 'route':[], 'completed':[]} for i in range(num_vehicles)]
    
    global_time, call_accumulator = 0, 0.0
    calls_per_min = base_calls_per_min * demand_multiplier
    
    times, sum_delays_s1, tail_sum_delays_s1, throughputs_s1, queue_lengths_s1 = [], [], [], [], []
    sum_delays_s2, tail_sum_delays_s2, throughputs_s2, queue_lengths_s2 = [], [], [], []
    occupancy_logs = []
    route_logs = []
    computation_time_logs = []
    total_vkt_s1, total_vkt_s2 = 0.0, 0.0

    while True:
        if global_time < sim_time:
            call_accumulator += calls_per_min * REOPT_INTERVAL
            num_new = int(call_accumulator)
            call_accumulator -= num_new
            
            reqs = []
            for _ in range(num_new):
                pu_cell, do_cell = sample_cell(), sample_cell()
                while do_cell == pu_cell: do_cell = sample_cell()
                reqs.append({
                    'call_min': round(global_time + random.random(), 2),
                    'pickup': (pu_cell[1], pu_cell[0]), 'dropoff': (do_cell[1], do_cell[0]),
                    'ideal_time': travel_time((pu_cell[1], pu_cell[0]), (do_cell[1], do_cell[0])),
                    'passenger_num': passenger_counter
                })
                passenger_counter += 1
            
            v_locs_s1 = {v['id']: v['loc'] for v in vehicles_s1}
            v_locs_s2 = {v['id']: v['loc'] for v in vehicles_s2}
            
            for req in reqs:
                travel_times_to_pickup = {
                    's1': min([manhattan_distance(loc, req['pickup']) for loc in v_locs_s1.values()]) / VEHICLE_SPEED,
                    's2': min([manhattan_distance(loc, req['pickup']) for loc in v_locs_s2.values()]) / VEHICLE_SPEED
                }
                passenger_travel_times[req['passenger_num']] = travel_times_to_pickup
                
                start_time_s1 = time.perf_counter()
                vehicles_s1 = assign_and_optimize_ALNS(vehicles_s1, req.copy(), 's1', global_time, 0.0, passenger_travel_times, vehicle_capacity, ideal_mode)
                end_time_s1 = time.perf_counter()
                computation_time_logs.append({
                    'Time': global_time, 'Seed': seed, 'Demand': f"demand_{demand_multiplier:g}x", 'Alpha': 0.0,
                    'Scenario': 'S1', 'ComputationTime': end_time_s1 - start_time_s1
                })

                start_time_s2 = time.perf_counter()
                vehicles_s2 = assign_and_optimize_ALNS(vehicles_s2, req.copy(), 's2', global_time, alpha, passenger_travel_times, vehicle_capacity, ideal_mode)
                end_time_s2 = time.perf_counter()
                computation_time_logs.append({
                    'Time': global_time, 'Seed': seed, 'Demand': f"demand_{demand_multiplier:g}x", 'Alpha': alpha, 
                    'Scenario': 'S2', 'ComputationTime': end_time_s2 - start_time_s2
                })
        
        for v in vehicles_s1:
            sim_route = simulate_dynamic_route(v['route'], global_time, v['loc'])
            exe, rem, loc, dist = update_executed_route(sim_route, global_time + REOPT_INTERVAL, global_time, v['loc'])
            v['completed'].extend(exe)
            v['loc'], v['route'] = loc, rem
            total_vkt_s1 += dist

        for v in vehicles_s2:
            sim_route = simulate_dynamic_route(v['route'], global_time, v['loc'])
            exe, rem, loc, dist = update_executed_route(sim_route, global_time + REOPT_INTERVAL, global_time, v['loc'])
            v['completed'].extend(exe)
            v['loc'], v['route'] = loc, rem
            total_vkt_s2 += dist

        for v in vehicles_s1:
            picked_up_passengers = {n['passenger_num'] for n in v['completed'] if n['type'] == 'pickup'}
            dropped_off_passengers = {n['passenger_num'] for n in v['completed'] if n['type'] == 'dropoff'}
            occupancy = len(picked_up_passengers - dropped_off_passengers)
            occupancy_logs.append({'Time': global_time, 'Seed': seed, 'Scenario': 'S1', 'VehicleID': v['id'], 'Occupancy': occupancy})
        for v in vehicles_s2:
            picked_up_passengers = {n['passenger_num'] for n in v['completed'] if n['type'] == 'pickup'}
            dropped_off_passengers = {n['passenger_num'] for n in v['completed'] if n['type'] == 'dropoff'}
            occupancy = len(picked_up_passengers - dropped_off_passengers)
            occupancy_logs.append({'Time': global_time, 'Seed': seed, 'Scenario': 'S2', 'VehicleID': v['id'], 'Occupancy': occupancy})

        for v in vehicles_s1:
            route_str = ' '.join([f"+{n['passenger_num']}" if n['type'] == 'pickup' else f"-{n['passenger_num']}" for n in v['route']])
            route_logs.append({'Time': global_time, 'Seed': seed, 'Scenario': 'S1', 'VehicleID': v['id'], 'Route': route_str})
        for v in vehicles_s2:
            route_str = ' '.join([f"+{n['passenger_num']}" if n['type'] == 'pickup' else f"-{n['passenger_num']}" for n in v['route']])
            route_logs.append({'Time': global_time, 'Seed': seed, 'Scenario': 'S2', 'VehicleID': v['id'], 'Route': route_str})

        s1_sum, s1_tail_sum = compute_metrics(vehicles_s1, 's1', ideal_mode, TAIL_PCT, passenger_travel_times)
        s2_sum, s2_tail_sum = compute_metrics(vehicles_s2, 's2', ideal_mode, TAIL_PCT, passenger_travel_times)
        
        throughputs_s1.append(len([n for v in vehicles_s1 for n in v['completed'] if n['type'] == 'dropoff']))
        throughputs_s2.append(len([n for v in vehicles_s2 for n in v['completed'] if n['type'] == 'dropoff']))
        queue_lengths_s1.append(sum(1 for v in vehicles_s1 for node in v['route'] if node['type'] == 'pickup'))
        queue_lengths_s2.append(sum(1 for v in vehicles_s2 for node in v['route'] if node['type'] == 'pickup'))
        
        sum_delays_s1.append(s1_sum)
        tail_sum_delays_s1.append(s1_tail_sum)
        sum_delays_s2.append(s2_sum)
        tail_sum_delays_s2.append(s2_tail_sum)
        times.append(global_time)

        global_time += REOPT_INTERVAL
        if global_time >= sim_time and all(len(v['route'])==0 for v in vehicles_s1) and all(len(v['route'])==0 for v in vehicles_s2): break

    def process_seed_results(vehicles, tag, passenger_travel_times):
        all_nodes = [node for v in vehicles for node in v['completed']]
        results = [res for n in all_nodes if n['type']=='dropoff' and (res := extract_result_fields(n, all_nodes, tag, passenger_travel_times))]
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    df_s1 = process_seed_results(vehicles_s1, 's1', passenger_travel_times)
    df_s2 = process_seed_results(vehicles_s2, 's2', passenger_travel_times)
    df_occupancy = pd.DataFrame(occupancy_logs)
    df_routes = pd.DataFrame(route_logs)
    df_computation_times = pd.DataFrame(computation_time_logs)
    
    df_trajectory = pd.DataFrame({
        'Time': times, 'S1_SumDelay': sum_delays_s1, 'S1_TailSumDelay': tail_sum_delays_s1,
        'S2_SumDelay': sum_delays_s2, 'S2_TailSumDelay': tail_sum_delays_s2,
        'S1_Throughput': throughputs_s1, 'S1_Queue': queue_lengths_s1,
        'S2_Throughput': throughputs_s2, 'S2_Queue': queue_lengths_s2, 'Seed': seed
    })

    summary_records = []
    delay_col = 'Delay' if ideal_mode == 1 else 'Delay2'
    for scenario_name, df, total_vkt in [('S1', df_s1, total_vkt_s1), ('S2', df_s2, total_vkt_s2)]:
        if not df.empty and delay_col in df.columns:
            summary_records.append({
                'Demand': f"demand_{demand_multiplier:g}x", 'Scenario': scenario_name, 'Seed': seed,
                'Alpha': alpha if scenario_name == 'S2' else np.nan, 'Avg_Delay': df[delay_col].mean(),
                'Gini_Delay': safe_gini(df, delay_col),
                'Tail_Delay': np.mean(np.sort(df[delay_col])[-max(1, int(len(df[delay_col])*TAIL_PCT)):]),
                'Avg_Detour': df['DetourFactor'].mean(),
                'Total_VKT': total_vkt
            })
    
    return summary_records, df_s1, df_s2, df_trajectory, df_occupancy, df_routes, df_computation_times, alpha, f"demand_{demand_multiplier:g}x"

# =============================================
# 5) Main Execution Logic
# =============================================
def main():
    sim_time, base_calls_per_min, num_vehicles, vehicle_capacity, num_seeds, ideal_mode, max_workers = get_parameters()

    params = {
        "sim_time": sim_time, "base_calls_per_min": base_calls_per_min,
        "num_vehicles": num_vehicles, "vehicle_capacity": vehicle_capacity,
        "num_seeds": num_seeds, "ideal_mode": ideal_mode,
        "start_alpha": START_ALPHA, "end_alpha": END_ALPHA, "step_alpha": STEP_ALPHA,
        "start_demand": START_DEMAND_MULTIPLIER, "end_demand": END_DEMAND_MULTIPLIER, "step_demand": STEP_DEMAND_MULTIPLIER,
        "service_time": SERVICE_TIME, "vehicle_speed": VEHICLE_SPEED,
        "reopt_interval": REOPT_INTERVAL, "tail_pct": TAIL_PCT
    }

    now_str_dir = datetime.datetime.now().strftime('sweep_%Y%m%d_%H%M%S')
    run_output_dir = os.path.join(BASE_OUTPUT_DIR, now_str_dir)
    detailed_results_dir = os.path.join(run_output_dir, "detailed_results")
    os.makedirs(detailed_results_dir, exist_ok=True)
    
    params_filename = os.path.join(run_output_dir, 'parameters.json')
    with open(params_filename, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    
    print(f"Starting full simulation sweep (using up to {max_workers} CPU cores)")
    print(f"Results will be saved to: {run_output_dir}")
    print(f"Parameters saved: {params_filename}")

    alpha_values = np.round(np.arange(START_ALPHA, END_ALPHA + STEP_ALPHA, STEP_ALPHA), 2)
    num_demand_steps = int(round((END_DEMAND_MULTIPLIER - START_DEMAND_MULTIPLIER) / STEP_DEMAND_MULTIPLIER)) + 1
    demand_multipliers = np.round(np.linspace(START_DEMAND_MULTIPLIER, END_DEMAND_MULTIPLIER, num_demand_steps), 2)

    tasks = []
    for alpha in alpha_values:
        for demand_multiplier in demand_multipliers:
            for seed in range(num_seeds):
                tasks.append((alpha, demand_multiplier, seed, sim_time, base_calls_per_min, num_vehicles, ideal_mode, vehicle_capacity))

    tradeoff_summary_data = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(run_simulation_task, tasks), total=len(tasks), desc="Total Progress"))

    print("\nAll simulations completed. Aggregating and saving results...")
    
    grouped_results = {}
    for summary_records, df_s1, df_s2, df_trajectory, df_occupancy, df_routes, df_computation_times, alpha, demand_name in results:
        tradeoff_summary_data.extend(summary_records)
        
        if (alpha, demand_name) not in grouped_results:
            grouped_results[(alpha, demand_name)] = {'s1': [], 's2': [], 'traj': [], 'occupancy': [], 'routes': [], 'comp_time': []}
        
        grouped_results[(alpha, demand_name)]['s1'].append(df_s1)
        grouped_results[(alpha, demand_name)]['s2'].append(df_s2)
        grouped_results[(alpha, demand_name)]['traj'].append(df_trajectory)
        grouped_results[(alpha, demand_name)]['occupancy'].append(df_occupancy)
        grouped_results[(alpha, demand_name)]['routes'].append(df_routes)
        grouped_results[(alpha, demand_name)]['comp_time'].append(df_computation_times)

    for (alpha, demand_name), data in tqdm(grouped_results.items(), desc="Saving detailed files"):
        alpha_dir = os.path.join(detailed_results_dir, f"alpha_{alpha:.1f}")
        os.makedirs(alpha_dir, exist_ok=True)

        with pd.ExcelWriter(os.path.join(alpha_dir, f'results_s1_{demand_name}.xlsx')) as writer:
            for i, df in enumerate(data['s1']): df.to_excel(writer, sheet_name=f'seed_{i}', index=False)
        
        with pd.ExcelWriter(os.path.join(alpha_dir, f'results_s2_{demand_name}.xlsx')) as writer:
            for i, df in enumerate(data['s2']): df.to_excel(writer, sheet_name=f'seed_{i}', index=False)
        
        combined_trajectory_df = pd.concat(data['traj'], ignore_index=True)
        csv_filename = f"time_series_data_{demand_name}.csv"
        trajectory_filepath = os.path.join(alpha_dir, csv_filename)
        combined_trajectory_df.to_csv(trajectory_filepath, index=False, encoding='utf-8-sig')

        combined_occupancy_df = pd.concat(data['occupancy'], ignore_index=True)
        csv_filename = f"occupancy_data_{demand_name}.csv"
        occupancy_filepath = os.path.join(alpha_dir, csv_filename)
        combined_occupancy_df.to_csv(occupancy_filepath, index=False, encoding='utf-8-sig')

        combined_routes_df = pd.concat(data['routes'], ignore_index=True)
        csv_filename = f"route_data_{demand_name}.csv"
        routes_filepath = os.path.join(alpha_dir, csv_filename)
        combined_routes_df.to_csv(routes_filepath, index=False, encoding='utf-8-sig')

        combined_comp_time_df = pd.concat(data['comp_time'], ignore_index=True)
        csv_filename_comp = f"computation_times_{demand_name}.csv"
        comp_time_filepath = os.path.join(alpha_dir, csv_filename_comp)
        combined_comp_time_df.to_csv(comp_time_filepath, index=False, encoding='utf-8-sig')

    summary_df = pd.DataFrame(tradeoff_summary_data)
    summary_filename = os.path.join(run_output_dir, '1_all_simulations_summary.csv')
    summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
    print(f"\nAll tasks completed. Final summary saved: {summary_filename}")

if __name__ == '__main__':
    main()
