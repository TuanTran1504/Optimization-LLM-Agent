from state import *
from utils import edges_to_distance_matrix
from extraction_tools import formalize_distance
import json
def evaluate(state: AgentState):
    print("---EVALUATE EXTRACTION---")

    router_instructions = """
    You are an expert at collecting information from user queries and determining the appropriate path.
    You are the second liner of solving the Optimization Problem like Traveling Salesman Problem
    You will examine the output of the node extractor to see whether there are still missing information to solve the problem or not

    
    Here are the information that have taken place:
    {information}

    For the information that you need:
    - If you dont have all those information, make sure to ask the user for more.
    - Starting node and Ending node are required to calculate.
    - If no constraint were found, confirm with the user. However, if they have stated in the conversation that they have no constraint, ignore the confirmation step.
    - If anything is confusing or vague, please confirm with the user by verification
    - If there are missing edges, please return true in need verification with the follow up question
    

    Respond in valid JSON format with these exact keys:
        "need_clarification": boolean,
        "question": "<question to ask the user to clarify the report scope>",
        "verification": "<verification message that we will start research>"
    If you need to ask a clarifying question, return:
        "need_clarification": true,
        "question": "<your clarifying question>",
        "verification": ""
    If you do not need to ask a clarifying question, return:
        "need_clarification": false,
        "question": "",
        "verification": "<acknowledgement message that you will now start research based on the provided information>"

    ONLY output verification if there are sufficient information
    For the verification message when no clarification is needed:
    - Acknowledge that you have sufficient information to proceed
    - Briefly summarize the key aspects of what you understand from their request
    - Confirm that you will now begin solve the problem
    - Keep the message concise and professional
    """
    
    structured_json_mode = llm.with_structured_output(ClarifyWithUser)
    route_question_msg = structured_json_mode.invoke(
        [SystemMessage(content=router_instructions), HumanMessage(content=str(state["brief"]))]
    )
    source = route_question_msg.need_clarification
    if source is False or source == "false":
        return  Command(
            goto="solver", 
            update={"messages": [AIMessage(content=route_question_msg.verification)]})   
    elif source is True or source == "true":
        return  Command(
            goto=END, 
            update={"messages": [AIMessage(content=route_question_msg.question)]})
  

def route_question(state: AgentState):
    print("---ROUTE QUESTION---")
    router_instructions = """
    You are an expert at collecting information from user queries and determining the appropriate path.
    You are the first liner of solving the Optimization Problem like Traveling Salesman Problem
    
    Here are the conversations that have taken place:
    {conversations}
    For the information that you might need:
    - All the constraint(time constraint, money constraint, penalty constraint), problem description
    - Starting nodes and Ending nodes are required to have.
    - The locations, nodes and distance between them or a table of coordinates in 2D if possible
    - If the users provide addresses, real locations, cities, you do not have to ask for the distance between those locations.
    - If you dont have all those information, make sure to ask the user for more.
    - If the user declare that there is no more information, do not ask for more
    
    

    If you need to ask a question, follow these guidelines:
    - Stricly limit to one question at a time
    - Make sure to ask for any constraint, special requirements if the customer have
    - Please approach the question with friendliness, understanding and willing to help the customers

    

    Respond in valid JSON format with these exact keys:
        "need_clarification": boolean,
        "question": "<question to ask the user to clarify the report scope>",
        "verification": "<Ask the user if they want to start the solving process>"
    If you need to ask a clarifying question, return:
        "need_clarification": true,
        "question": "<your clarifying question>",
        "verification": ""
    If you do not need to ask a clarifying question, return:
        "need_clarification": false,
        "question": "",
        "verification": "<Ask the user if they want to start the solving process>"

    For the verification message when no clarification is needed:
    - Acknowledge that you have sufficient information to proceed
    - Briefly summarize the key aspects of what you understand from their request
    - Ask the user if they want to start the solving process
    - Keep the message concise and professional
    """
    def deep_merge_constraints(old: dict, new: dict) -> dict:
        """Recursively merge constraint dictionaries. Null or empty values in new do NOT overwrite old."""
        merged = old.copy()
        for key, value in new.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = deep_merge_constraints(merged[key], value)
            elif value not in [None, [], {}, ""]:
                merged[key] = value
            # if value is None, empty list, or empty dict, keep old value
        return merged


    current_node=state.get("current_node")
    if current_node:
        if current_node[0] == "route_question":
            pass
        else:
            return Command(goto=str(current_node[0]))
        
    state["current_node"] = ["route_question"]

    user_content = (
        f"Conversations:\n{state['messages']}"
    )
    clarify_json_mode = llm.with_structured_output(ClarifyWithUser)
    route_question_msg = clarify_json_mode.invoke(
        [SystemMessage(content=router_instructions), HumanMessage(content=user_content)]
    )

    existing = state.get("constraints", {})
    if hasattr(existing, "model_dump"):
        existing_dict = existing.model_dump()
    elif hasattr(existing, "dict"):
        existing_dict = existing.dict()
    else:
        existing_dict = existing
    new_msg = state["messages"][-1]

    prompt = f"""
    You are a constraint updater.
    Here is the current constraint JSON:
    {json.dumps(existing_dict, indent=2)}

    User message:
    \"\"\"{new_msg}\"\"\"

    Update only the fields the user changes, adds, or removes.
    Keep all other values identical.
    Make sure the name is consistent
    All times in time_windows.start or time_windows.end MUST be strings in HH:MM or 24-hour format.
    Do NOT return numeric minute values.
    Return the FULL updated JSON following this schema:
    {{
      "visit_all_nodes": true/false/null,
      "tour_closed": true/false/null,
      "start_node": string/null,
      "end_node": string/null,
      "travel_mode": string/null,
      "objective": string/null,
      "time_windows": object/null,
      "capacity": object/null,
      "nodes": object/null,
      "other_constraints": object/null,
      "avg_spd": float/null,
      "precedence_pairs": list/null,
      forbidden_edges": list/null
    }}
    """

    constraints_json_mode = llm.with_structured_output(Constraints, method="function_calling")
    resp = constraints_json_mode.invoke([SystemMessage(content=prompt)])
    if isinstance(resp, str):
        updated = json.loads(resp)
    else:
        updated = resp.model_dump()

    merged_constraints = deep_merge_constraints(existing_dict, updated)
   
    state["constraints"] = merged_constraints
    source = route_question_msg.need_clarification

    
    if source is False or source == "false":
        return  Command(
            goto=END, 
            update={"verification": [route_question_msg.verification], "messages":[AIMessage(content=route_question_msg.verification)], "current_node": ["verification"],"constraints": merged_constraints})   
    elif source is True or source == "true":
        return  Command(
            goto=END, 
            update={"messages": [AIMessage(content=route_question_msg.question)], "current_node": ["route_question"],"constraints": merged_constraints})


def node_extraction(state: AgentState):
    print("START EXTRACTING NODES")
    messages = state.get("messages", [])
    last_user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    current_node = state.get("nodes", {})
    current_matrix = state.get("matrix", {})
    node_matrix = f"Nodes: {current_node}\nMatrix: {current_matrix}"
    decision_prompt = """Your job is to decide whether there should be an update in the nodes, through the last conversation
    If the user mention anything related to the nodes about their distance or their names, you need to review with the current distance matrix   
    Here are the current nodes
    {currentnodes}
    Answer in this schema:
        need_update: <true or false>

    Return true if there is a need to update or add to the node matrix and false if there is nothing to change
    However, if you dont see the current node or an empty matrix, in that case, you should always return true
    """
    decision_prompt = decision_prompt.format(currentnodes=node_matrix)
    extract_decision_llm = llm.with_structured_output(Extract_decision)
    decision = extract_decision_llm.invoke([SystemMessage(content=decision_prompt), HumanMessage(content=last_user_msg)])


    if str(getattr(decision, "need_update", "false")).lower() == "true" or node_matrix is None:
        command_prompt = """
        You are given a conversation between a user and a system. 
        Your task is to transform this conversation into a single concise command that captures the distance, travel time, time cost or coordinates. 
            - If the conversation mentions specific locations (nodes) with no distance given, rewrite it as a command of the form:
            "Identify the pairwise <distances or travel time> between <Location1>, <Location2>, …"
            - Ignore system confirmations or filler dialogue. 
            - If the conversation includes coordinates of nodes, you need to pass it to your command in this form:
            "Identify the distance the given nodes coordinates: <node1>: <longtitude>, <latitude>"
            - If the conversation includes travel time, distance between nodes, you need to pass it to your command in this form:
            "Identify the distance between <node1> - <node2>: <distance/travel time/time cost>"
        Do not create any other command that is outside of your scope
        Do not invent distances. Only extract the locations explicitly mentioned by the user.
        Output only the final command, nothing else.   
        Do not send command to collect things other than distance, travel time, time cost or coordinates.
        Please use the extract node name from the current nodes

        Here are the current nodes and maybe distance matrix (this could be empty)
        {currentnodes}  
        Provide only the command
        """
        user_content = (
            f"Conversations:\n{messages}"
        )
        command_prompt = command_prompt.format(currentnodes=node_matrix)
        resp = llm.invoke([SystemMessage(content=command_prompt),HumanMessage(content=user_content)])
        distance_matrix = formalize_distance(resp.content)

        matrix= edges_to_distance_matrix(distance_matrix)
        print(matrix)
        return  Command(
                    goto="briefing", 
                    update={
                        "nodes": matrix["nodes"],
                        "distance_matrix": matrix["matrix"],
                        "messages": [AIMessage(content=f"Updated nodes and distances for {matrix['nodes']}")]
                })
    else:
        return Command(goto="briefing", update={"messages": [AIMessage(content="No node updates detected.")]})


def briefing(state: AgentState):
    print("START BRIEFING")
    state["current_node"]=["briefing"]
    prompt = """You are an agent specialized in summarize the context of the conversation.
    You are required to provide a brief summary of the requirements from the users about solving the TSP problem
    If there are many TSP scenario, choose the most current one. If you are not sure, return a clarifying question in the below format
    Respond in valid JSON format with these exact keys:
        "summary": "<Your main summary of the requirements>"

    Here are the conversations:
    {messages}  
    Provide only the normalize version
    """
    user_content = (
        f"Conversations:\n{state['messages']}, Constraints:\n{state['constraints']}"
    )
    
    resp = llm.invoke([SystemMessage(content=prompt),HumanMessage(content=user_content)])
    structured_json_mode = llm.with_structured_output(Briefing)
    route_question_msg = structured_json_mode.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=user_content)]
    )
    return  Command(
            goto="evaluate", 
            update={"brief": [AIMessage(content=route_question_msg.summary)], "current_node": ["briefing"], "summary": route_question_msg.summary })  
    

def greeting(state: AgentState):
    current_node = state.get("current_node", [])
    if current_node:
        if current_node[0] != "greeting":
            return Command(goto=current_node[0])
    print("Greeting")
    router_instructions = """
    You are an expert at greeting customer with well mannered conversation
    You will have to be able to provide the users what they need to know.
    You are also guide the users to different agent according to their needs
    Currently there is one agen which is TSP agent which is capable of solving any TSP problems
    
    Here are the conversations that have taken place:
    {conversation}

    
    Your action is depending on what the customer ask and also the conversation flow.
    - If the user did not ask anything related to TSP problem, give them a direct answer
    - If the user ask about anything related to TSP problem, you have to pass the conversation to the TSP planner
    

    Respond in valid JSON format with these exact keys:
        "routing": boolean,
        "answer": "<Answer within your scope>",
        
    If you need to pass to the TSP agent, return:
        "routing": true,
        "answer": "",
    If you do not need to pass to the TSP agent, return:
        "routing": false,
        "answer": "<Your answer to the user question>",
    """
    structured_json_mode = llm.with_structured_output(Greeting)
    route_question_msg = structured_json_mode.invoke(
        [SystemMessage(content=router_instructions), HumanMessage(content=str(state["messages"]))]
    )
    source = route_question_msg.routing
    if source is False or source == "false":
        return  Command(
            goto=END, 
            update={"messages": [AIMessage(content=route_question_msg.answer)], "current_node":["greeting"]})   
    elif source is True or source == "true":
        return  Command(
            goto="route_question", 
            update={"current_node": ["route_question"]})
    
def verification(state: AgentState):
    print("--VERIFICATION--")
    verification = state.get("verification", {})
    user_msg = state.get("messages", {})
    router_instructions = """
    You are an expert at verify customer agreement to start solving.
    You will be giving a question and an answer from user. You will carefully decide whether the user has allowed to continue or not
    Please strictly follow the user command
    Here is the verification question:
    {verification}

    Respond in valid JSON format with these exact keys:
        "decision": <true or false>
    true: User verify to continue
    false: User not verify to continue
    
    Here is the answer:
    {user_answer}
    """
    router_instructions =router_instructions.format(verification=f"Verification:{verification}", user_answer=f"User: {user_msg[-1].content}")
    structured_json_mode = llm.with_structured_output(Verification)
    route_question_msg = structured_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
    )
    
    source = route_question_msg.decision
    
    if source is False or source == "false":
        return  Command(
            goto="route_question", update={"current_node": ["route_question"]})
    elif source is True or source == "true":
        return  Command(
            goto="node_extraction",
            update={"current_node":["node_extraction"]})