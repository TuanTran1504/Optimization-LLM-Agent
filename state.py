from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Optional, Annotated, List ,Sequence
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

from typing import Dict, List, Optional, Union, Tuple
from langgraph.types import Command
from langgraph.graph import END, START
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from typing import Dict
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4.1-mini",temperature=0, api_key=api_key)


# llm = ChatOllama(model="llama3", temperature=0,base_url="http://127.0.0.1:11434")
class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions."""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )
class Solver(BaseModel):
    """Schema for solver."""
    answer: str = Field(
        description="An explaination of the solution to the user",
    )
class Briefing(BaseModel):
    """Schema for briefing the problem"""
    summary: str = Field(
        description="Summary of the requirements",
    )
class Extract_decision(BaseModel):
    """Schema for extract nodes decision"""
    need_update: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    
class Verification(BaseModel):
    """Schema for user verification decisions during scoping phase."""
    decision: bool = Field(
        description="Decide to continue or keep taking question",
    )
    
class Greeting(BaseModel):
    routing: bool = Field(
        description="Whether you need to pass the conversation to TSP agent or not.",
    )
    answer: str = Field(
        description="A direct answer",
    )
class CapacityConstraint(BaseModel):
    vehicle_capacity: Optional[int] = None
    demands: Optional[Dict[str, int]] = None


class TimeWindow(BaseModel):
    start: Optional[str] = Field(None, description="Earliest allowed arrival time (HH:MM)")
    end: Optional[str] = Field(None, description="Latest allowed arrival time (HH:MM)")

class TimeWindowConstraint(BaseModel):
    windows: Optional[Dict[str, TimeWindow]] = Field(
        default=None,
        description="Mapping from node → {start, end} time window"
    )
class Constraints(BaseModel):
    """Schema for optimization constraints, normalized for evaluation."""

    # Simple flags
    visit_all_nodes: Optional[bool] = Field(
        default=None, description="Whether all nodes must be visited"
    )
    tour_closed: Optional[bool] = Field(
        default=None, description="Whether the tour starts and ends at the same node"
    )
    
    # Routing
    start_node: Optional[str] = Field(default=None, description="Starting location")
    end_node: Optional[str] = Field(default=None, description="Ending location")
    travel_mode: Optional[str] = Field(
        default=None, description="driving, walk, air"
    )
    objective: Optional[str] = Field(
        default=None,
        description="Optimization objective, e.g. 'min_distance', 'min_time', 'max_value'",
    )

    # Structured sub-objects
    capacity: Optional[CapacityConstraint] = None
    time_windows: Optional[TimeWindowConstraint] = None

    # Future extensions (VRP, scheduling, knapsack etc.)
    
    other_constraints: Optional[Dict[str, Union[str, int, float, dict, list, None]]]= Field(
        default=None,
        description="For unstructured or future constraint types not yet modeled"
    )
    nodes: Optional[List[str]] = Field(
        default=None,
        description="List of all node names or locations in the problem",
    )
    distance_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="2D distance or cost matrix aligned with the node list order",)
    avg_spd: Optional[float] = Field(
        default=None,
        description="Average vehicle speed in km/h (optional)"
    )
    precedence_pairs: Optional[List[List[str]]] = Field(
        default=None,
        description=(
            "List of ordered pairs (A, B) meaning node A must be visited before node B. "
            "Used in scheduling, pickup-before-delivery, or dependency routing scenarios."
        )
    )

    forbidden_edges: Optional[List[List[str]]] = Field(
        default=None,
        description=(
            "List of disallowed direct transitions between nodes, e.g. ('A', 'C') means A→C cannot be used."
        )
    )
class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """
    answer: Optional[str]
    brief: Optional[str] 
    verification: Optional[str] 
    extracted_data: Optional[str]
    current_node: Optional[str]
    constraints: Optional[str] 
    nodes: Optional[str] = ""
    distance_matrix: Optional[str] = ""

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