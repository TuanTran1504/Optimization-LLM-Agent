import json
import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from datetime import datetime
from state import *
import re
from pprint import pprint
import unicodedata
from typing import Optional, List, Tuple
import difflib

def find_best_match(name, node_list):
    norm_name = normalize_location(name)
    norm_nodes = [normalize_location(n) for n in node_list]
    matches = difflib.get_close_matches(norm_name, norm_nodes, n=1, cutoff=0.7)
    if matches:
        return norm_nodes.index(matches[0])
    return None

def normalize_location(name: str) -> str:
    """Normalize a location string for consistent matching."""
    if not name or not isinstance(name, str):
        return ""  # or return None if you prefer explicit nulls

    name = name.lower().strip()
    name = re.sub(r"[.,]", "", name)
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"\s+", " ", name)
    name = name.replace("d c", "dc")
    return name.strip()

def parse_time_to_minutes(t: Optional[str]) -> Optional[int]:
    """Convert 'HH:MM' string to minutes since midnight."""
    if t is None:
        return None
    h, m = map(int, t.split(":"))
    return h * 60 + m

def formalize_time_windows(constraints: Constraints, node_order: Optional[List[str]] = None):
    time_dict = {}
    if constraints.time_windows and constraints.time_windows.windows:
        time_dict = {normalize_location(k): v for k, v in constraints.time_windows.windows.items()}

    cap_demands = {}
    if constraints.capacity and constraints.capacity.demands:
        cap_demands = {normalize_location(k): v for k, v in constraints.capacity.demands.items()}

    constraint_nodes = [normalize_location(n) for n in (constraints.nodes or [])]
    print(constraint_nodes)
    print(node_order)
    if node_order:
        ordered_nodes = [normalize_location(n) for n in node_order]
    elif constraint_nodes:
        ordered_nodes = constraint_nodes
    elif time_dict:
        ordered_nodes = list(time_dict.keys())
    elif cap_demands:
        ordered_nodes = list(cap_demands.keys())
    else:
        ordered_nodes = []

    def parse_time_to_minutes(t: Optional[str]) -> Optional[int]:
        if not t:
            return None
        h, m = map(int, t.split(":"))
        return h * 60 + m

    time_windows_numeric = []
    if not time_dict:
        time_windows_numeric = [(0, 24 * 60)] * len(ordered_nodes)
    else:
        for node in ordered_nodes:
            if node in time_dict:
                tw = time_dict[node]
                start = parse_time_to_minutes(tw.start) or 0
                end = parse_time_to_minutes(tw.end) or 24 * 60
            else:
                start, end = 0, 24 * 60
            time_windows_numeric.append((start, end))

    demands_numeric, vehicle_capacity = None, None
    if constraints.capacity:
        vehicle_capacity = constraints.capacity.vehicle_capacity
        if cap_demands:
            demands_numeric = [cap_demands.get(node, 0) for node in ordered_nodes]
        else:
            demands_numeric = [0] * len(ordered_nodes)

    start_node = normalize_location(constraints.start_node) if constraints.start_node else None
    end_node = normalize_location(constraints.end_node) if constraints.end_node else None
    avg_spd = getattr(constraints, "avg_spd", None)
    return time_windows_numeric, demands_numeric, vehicle_capacity, start_node, end_node, avg_spd

def edges_to_distance_matrix(data):
    """
    Convert a list or dict of edges into a symmetric distance matrix.
    Accepts:
        - a list of edges: [{"node1": ..., "node2": ..., "distance": ...}]
        - a dict containing {"edges": [...]}
        - a JSON string of either form
    """
    # 🔹 Step 1: Normalize input
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string for edges_to_distance_matrix")

    # 🔹 Step 2: Extract edges list
    if isinstance(data, dict) and "edges" in data:
        edges = data["edges"]
    elif isinstance(data, list):
        edges = data
    else:
        raise TypeError(f"Invalid input type for edges_to_distance_matrix: {type(data)}")

    # 🔹 Step 3: Build node index and matrix
    nodes = sorted(set([e["node1"] for e in edges] + [e["node2"] for e in edges]))
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    D = np.zeros((n, n))
    for e in edges:
        i, j = node_index[e["node1"]], node_index[e["node2"]]
        D[i][j] = e["distance"]
        D[j][i] = e["distance"]  # assume symmetric unless told otherwise

    return {"nodes": nodes, "matrix": D.tolist()}


def search(
    matrix: list,
    start: int = None,
    end: int = None,
    closed_loop: bool = False,
    depot: int = None,
    num_vehicle: int = 1,
    time_windows: list | None = None,
    avg_spd: int | None = None,
    demands: list | None = None,
    vehicle_capacity: int | None = None,
    forbidden_edges=None,   
    precedence_pairs=None
):
    # --- Debug types to be 100% sure ---
    print(
        "DEBUG:",
        len(matrix), type(len(matrix)),
        num_vehicle, type(num_vehicle),
        start, type(start),
        end, type(end),
    )

    # --- Manager: open vs closed route ---
    num_nodes = int(len(matrix))
    num_vehicle = int(num_vehicle)

    if not closed_loop and start is not None and end is not None:
        # OPEN route (start != end): starts/ends MUST be lists of ints
        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicle, [int(start)], [int(end)])
    elif closed_loop and depot is not None:
        # CLOSED loop (start == end): depot is a single int
        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicle, int(depot))
    else:
        raise ValueError("Provide depot for closed loop OR start & end for open route. "
                         f"closed_loop={closed_loop}, start={start}, end={end}, depot={depot}")
    
    routing = pywrapcp.RoutingModel(manager)

    # --- Distance callback MUST return int ---
    def distance_callback(from_index, to_index):
        fn = manager.IndexToNode(from_index)
        tn = manager.IndexToNode(to_index)
        # Cast to int to satisfy OR-Tools (non-negative integers)
        return int(round(float(matrix[fn][tn])))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # --- Capacity (optional) ---
    if demands is not None and vehicle_capacity is not None:
        def demand_callback(from_index):
            return int(demands[manager.IndexToNode(from_index)])
        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx,
            0,  # slack
            [int(vehicle_capacity)] * num_vehicle,
            True,
            "Capacity",
        )
    # Forbidden edges
    if forbidden_edges:
        for (i, j) in forbidden_edges:
            fi = manager.NodeToIndex(int(i))
            tj = manager.NodeToIndex(int(j))
            routing.solver().Add(routing.NextVar(fi) != tj) 
    # --- Time windows (optional) ---
    if time_windows:
        # If needed, pad to num_nodes so every node has a window
        if len(time_windows) < num_nodes:
            time_windows = list(time_windows) + [(0, 24*60)] * (num_nodes - len(time_windows))

        AVERAGE_SPEED = 60 if avg_spd is None else avg_spd  # miles/hour
        SERVICE_TIME_H = 0.25  # hours per stop

        def time_callback(from_index, to_index):
            fn = manager.IndexToNode(from_index)
            tn = manager.IndexToNode(to_index)
            travel_hours = float(matrix[fn][tn]) / float(AVERAGE_SPEED)
            minutes = (travel_hours + SERVICE_TIME_H) * 60.0
            return int(round(minutes))
            
        time_cb_idx = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_cb_idx,
            30,         # waiting/slack in minutes
            24 * 60,    # max horizon in minutes
            False,      # don't force start at time 0
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")
        for node, (open_t, close_t) in enumerate(time_windows[:num_nodes]):
            idx = manager.NodeToIndex(node)
            if idx == -1:
                continue
            time_dim.CumulVar(idx).SetRange(int(open_t), int(close_t))
        
        # Help the solver: minimize start/end 
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(0)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(0)))

    #Precedence pairs
    if precedence_pairs and time_windows:
        time_dim = routing.GetDimensionOrDie("Time")
        for (before, after) in precedence_pairs:
            bi = manager.NodeToIndex(int(before))
            aj = manager.NodeToIndex(int(after))
            routing.solver().Add(time_dim.CumulVar(bi) <= time_dim.CumulVar(aj))
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
 


    print("Solving...")
    solution = routing.SolveWithParameters(params)
    if not solution:
        print("⚠️ No feasible route found, returning None.")
        return None

    # --- Extract route safely ---
    route = []
    index = routing.Start(0)
    visited_guard = 0
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
        visited_guard += 1
        if visited_guard > num_nodes + 5:
            # Safety guard to avoid accidental infinite loops in extraction
            break
    route.append(manager.IndexToNode(index))
    print("Route (node ids):", route)

    return {
        "route_node_indices": route,
        "total_cost": solution.ObjectiveValue(),
    }

# distance_matrix = [
#     [0.0, 1364.0, 214.0, 308.0, 439.0],
#     [1364.0, 0.0, 1151.0, 1064.0, 920.0],
#     [214.0, 1151.0, 0.0, 94.0, 226.0],
#     [308.0, 1064.0, 94.0, 0.0, 152.0],
#     [439.0, 920.0, 226.0, 152.0, 0.0]
# ]

# time_windows = [
#     (0, 1440),    # DepotStart: full day
#     (0, 1440),    # Stop A
#     (0, 1440),    # Stop B
#     (840, 1440),  # Stop C after 2 PM
#     (900, 1440)     # DepotEnd by midnight
# ]

# demands = [0, 50, 10, -30, 0]
# vehicle_capacity = 200

# # --- Forbidden edges and precedence ---
# forbidden_edges = [(1, 4)]  # Can't go directly from A -> End
# precedence_pairs = [(2, 1)] # B must come before BA

# # --- Run the solver ---
# result = search(
#     matrix=distance_matrix,
#     start=0,
#     end=4,
#     avg_spd= 6000,
#     time_windows=time_windows,
#     demands=demands,
#     vehicle_capacity=vehicle_capacity,
#     forbidden_edges=forbidden_edges,
#     precedence_pairs=precedence_pairs
# )

# print("\n=== FINAL RESULT ===")
# print(result)

