from langchain_openai import ChatOpenAI
import os
from .tool_call import *
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import json
load_dotenv()

tools = [direction_search ,think_tool, haversine_distance, node_extraction, euclidean_distance]
tools_by_name = {tool.name: tool for tool in tools}

api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4.1-mini",temperature=0, api_key=api_key)
model_with_tools = llm.bind_tools(tools)
def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    print("Calling Agent")
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }
def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses and
    returns updated state with structured tool execution results.
    """
    print("Using Tools")
    tool_calls = state["researcher_messages"][-1].tool_calls
    print(tool_calls)

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])

        # Ensure the output is JSON-serializable string for ToolMessage
        if isinstance(result, (dict, list)):
            content = json.dumps(result, indent=2)
        else:
            content = str(result)

        observations.append(content)

    # Build tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # Capture any specific results (e.g., node_extraction outputs)
    notes = [
        output
        for output in tool_outputs
        if output.name == "node_extraction"
    ]
    return {
        "researcher_messages": tool_outputs,  # all tool messages
        "raw_notes": notes                    # only node_extraction outputs
    }

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"

def formalize_distance(query: str):
    agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    # Add nodes to the graph
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)


    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node", # Continue research loop
            "compress_research": END, # Provide final answer
        },
    )
    agent_builder.add_edge("tool_node", "llm_call")
    # Compile the agent
    researcher_agent = agent_builder.compile()
 
    result = researcher_agent.invoke({"researcher_messages": [HumanMessage(content=f"{query}.")]})
    data = json.loads(result["raw_notes"][-1].content)
 
    return data

# result= formalize_distance(query="Identify the distance between Alpha Base - Bravo Outpost: 5 km, Alpha Base - Charlie Station: 6 km, Alpha Base - Delta Hub: 10 km, Alpha Base - Echo Camp: 12 km, Bravo Outpost - Charlie Station: 3 km, Bravo Outpost - Delta Hub: 6 km, Bravo Outpost - Echo Camp: 8 km, Charlie Station - Delta Hub: 4 km, Charlie Station - Echo Camp: 6 km, Delta Hub - Echo Camp: 5 km")