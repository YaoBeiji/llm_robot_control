from typing import List, Dict
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

class MultiAgentState(BaseModel):
    messages: List[BaseMessage]
    memory: Dict[str, List[BaseMessage]] = {}

class State(TypedDict):
    messages: Annotated[list, add_messages]