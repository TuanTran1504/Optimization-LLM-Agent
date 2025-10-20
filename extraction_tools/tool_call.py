import googlemaps
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from datetime import datetime
from typing_extensions import Annotated, List, Literal
from .prompt import *
import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, field_validator
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict
import math
from langchain_openai import ChatOpenAI
import os
from typing import Optional, List
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import List, Optional, Tuple, Dict

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4.1-mini",temperature=0, api_key=api_key)
class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.
    
    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    nodes: str
    raw_notes: Annotated[List[str], operator.add]
class Edge(BaseModel):
    node1: str
    node2: str
    distance: Optional[int] = None


class NormalizedGraph(BaseModel):
    edges: List[Edge]


class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

class CityCoordinates(BaseModel):
    coords: Dict[str, List[float]]

    @field_validator('coords')
    def validate_keys_and_coords(cls, v):
        if not all(isinstance(k, str) and k.strip() for k in v.keys()):
            raise ValueError('All keys must be non-empty strings')
        for coord in v.values():
            if len(coord) != 2:
                raise ValueError(f'Coordinates must be lists of exactly two floats: {coord}')
            lat, lon = coord
            if not -90 <= lat <= 90:
                raise ValueError(f'Latitude {lat} must be between -90 and 90 degrees')
            if not -180 <= lon <= 180:
                raise ValueError(f'Longitude {lon} must be between -180 and 180 degrees')
        return v

    class Config:
        json_schema_extra = {
            "required": ["coords"]
        }
    


@tool(parse_docstring=True)
def direction_search(
    start: str,
    end: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    mode: str = "driving",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        start: Start location
        end: Destination location
        max_results: Maximum number of results to return
        mode: Travel mode by ('driving', 'walk', 'air')

    Returns:
        Formatted string of distance or duration between locations
    """
    # Execute search for single query
    key = os.getenv("GOOGLE_API_KEY")
    gmaps = googlemaps.Client(key=key)
    # Request directions via public transit
    now = datetime.now()
    directions_result = gmaps.directions(start,
                                        end,
                                        mode=mode,
                                        departure_time=now)

    leg = directions_result[0]['legs'][0]
    # Format output for consumption
    return {f"{start} - {end}": leg.get("distance", leg['duration'])["text"]}


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """
    Tool for strategic reflection on research progress and decision-making.
    
    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the workflow for quality decision-making.
    
    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough information of the nodes distance (including all the pairwise distances if no constraint is specified)
    - Before moving on: Can I provide a complete answer now?
    
    CAUTIONS:
    -Stop immediately if you found all the information to answer the initial query 
    
    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Strategic decision - Should I continue to use tools or provide my answer?

    Args:
        reflection: Your detailed reflection on searching and extraction progress, findings, gaps, and next steps


    Returns:
        str: A confirmation message recording the reflection.
    """
    return f"Reflection recorded: {reflection}"


@tool(parse_docstring=True)
def node_extraction(query: str):
    """
    Tool for collecting coordinates of the nodes from conversation.

    Args:
        query: The content of user conversation that mentioned the distance or travel time between nodes.

    Returns:
        dict: Dictionary of pairwise distnance.
    """
    print("START EXTRACTING NODES")
    prompt = """
    You are a distance normalization engine. 
    Your task is to extract all pairwise node distances from text in a consistent schema.

    Guidelines:
    - Identify all unique nodes (cities, stations, or locations).
    - Include **all** pairwise combinations between them, even if no distance is stated.
    - Convert spelled-out numbers to integers (e.g., “eight hundred seventy-eight” → 878).
    - If distance is missing or uncertain, set it to null.
    - If units are mentioned (e.g., km, miles, minutes), **ignore them** and keep numeric only.
    - Return each node as a short city name without state suffixes or punctuation
    - Do not hallucinate nodes not in the text.
    - Output must match the schema exactly: NormalizedGraph.
    """

    llm_with_schema = llm.with_structured_output(NormalizedGraph)

    # Invoke structured LLM call
    response = llm_with_schema.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ])
    return response.model_dump()
@tool(parse_docstring=True)
def haversine_distance(
    name1: str,
    name2: str,
    start_coords: Annotated[list, "Starting point as [lat, lon]"],
    end_coords: Annotated[list, "Ending point as [lat, lon]"],
    unit: str = "km"
) -> Dict[str, float]:
    """
    Calculate the great-circle distance between two points on Earth 
    using the Haversine formula.

    Args:
        name1: Name of starting node.
        name2: Name of ending node
        start_coords: List of [latitude, longitude] for the starting point.
                      Latitude must be between -90 and 90, longitude between -180 and 180.
        end_coords: List of [latitude, longitude] for the ending point.
                    Latitude must be between -90 and 90, longitude between -180 and 180.
        unit: The unit of distance to return ("km" for kilometers, "mi" for miles). Default = "km".

    Returns:
        dict: {"distance": <value>, "unit": "km" or "mi"}
    """
    print("Haversine")
    # Validate inputs
    for lat, lon in [start_coords, end_coords]:
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude {lat}, must be between -90 and 90")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude {lon}, must be between -180 and 180")

    # Convert degrees to radians
    lat1, lon1 = map(math.radians, start_coords)
    lat2, lon2 = map(math.radians, end_coords)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Earth radius
    R = 6371 if unit == "km" else 3958.8  # miles if requested
    distance = R * c
    return {f"{name1} - {name2}": round(distance, 2), "unit": unit}


@tool(parse_docstring=True)
def euclidean_distance(
    name1: str,
    name2: str,
    start_coords: Annotated[list, "Starting point as [x, y] in a local Cartesian system"],
    end_coords: Annotated[list, "Ending point as [x, y] in a local Cartesian system"],
    unit: str = "m"
) -> Dict[str, float]:
    """
    Calculate the straight-line (Euclidean) distance between two points
    in a planar or local coordinate system.

    Args:
        name1: Name of starting node.
        name2: Name of ending node.
        start_coords: [x, y] position of the starting point (units: meters, millimeters, pixels, etc.).
        end_coords: [x, y] position of the ending point.
        unit: The unit of the returned distance ("m", "mm", "km", etc.). Default = "m".

    Returns:
        dict: { "<name1> - <name2>": <distance>, "unit": unit }

    Notes:
        - This function is used for local coordinate systems where no latitude/longitude constraints apply.
        - Coordinates can be negative or exceed Earth coordinate bounds.
        - Example usage: CNC toolpath optimization, warehouse robot routing, PCB drilling.
    """

    # Validate coordinate format
    if not (isinstance(start_coords, (list, tuple)) and isinstance(end_coords, (list, tuple))):
        raise TypeError("Coordinates must be lists or tuples [x, y].")
    if len(start_coords) != 2 or len(end_coords) != 2:
        raise ValueError("Each coordinate must have exactly two elements [x, y].")

    x1, y1 = start_coords
    x2, y2 = end_coords

    # Compute Euclidean distance
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return {f"{name1} - {name2}": round(distance, 3), "unit": unit}