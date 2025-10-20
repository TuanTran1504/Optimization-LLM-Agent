from state import *
from utils import *
def solver(state: AgentState):
    print("---GENERATING FINAL ANSWER---")

    router_instructions = """
    You are an expert at solving Travelling Salesman Problem
    Your job is to explain the solution to the user
    {solution}
    
    Respond in valid JSON format with these exact keys:
        "answer": "<your explaination of the solution>"

    """
    
    matrix = state["distance_matrix"]
    constraints = state["constraints"]
    time_window = constraints.get("time_windows", None)
    nodes = [normalize_location(n) for n in state["nodes"]]
    time_window, demands, capacity, start_node, end_node, avg_spd = formalize_time_windows(Constraints(**constraints), nodes)
    start_index = find_best_match(start_node, nodes)
    end_index = find_best_match(end_node, nodes)
    if start_index is None or end_index is None:
        return  Command(
            goto=END, 
            update={"messages": [AIMessage(content=f"Please provide the start node and end node")],
                    "current_node": ["route_question"],
                    "constraints": state.get("constraints", {})})


    print(matrix, time_window, nodes, demands, capacity, start_index, end_index)

    is_closed = (start_node == end_node)

    # Prepare base arguments (always required)
    search_args = {
        "matrix": matrix,
        "num_vehicle": 1,  # or make this dynamic later
    }

    # --- Handle routing topology ---
    if is_closed:
        search_args["depot"] = start_index
        search_args["closed_loop"] = True
    else:
        search_args["start"] = start_index
        search_args["end"] = end_index
        search_args["closed_loop"] = False

    # --- Add optional parameters only if they exist ---
    if time_window:
        search_args["time_windows"] = time_window
    if demands is not None and capacity is not None:
        search_args["demands"] = demands
        search_args["vehicle_capacity"] = capacity
    if avg_spd is not None:
        search_args["avg_spd"] = avg_spd

    # --- Execute search ---
    result = search(**search_args)

    if result is None:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content="No viable solution found, please change your constraints")],
                    "current_node": ["route_question"],
                    "constraints": state.get("constraints", {})})
    
    router_instructions =router_instructions.format(solution=f"Verification:{result}")
    structured_json_mode = llm.with_structured_output(Solver)
    route_question_msg = structured_json_mode.invoke(
        [SystemMessage(content=router_instructions), HumanMessage(content=" ".join([str(state.get("brief", "")), str(state["nodes"]),str(state["constraints"])]))]
    )
    print(route_question_msg.answer)
    return  Command(
            goto=END, 
            update={"messages": [AIMessage(content=route_question_msg.answer)],
                    "current_node": ["route_question"],
                    "constraints": state.get("constraints", {})})
                     
 