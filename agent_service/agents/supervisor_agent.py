from langgraph.types import Send
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from . import ChatAgent, RobotAgent, VisionAgent, PlanAgent,InteractionAgent
from agent_service.memory import MultiAgentState ,State
from agent_service.config import get_llm
from agent_service.tools.config_load import ConfigLoader
from langgraph.graph import StateGraph, START,END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

memory_saver = MemorySaver()

def create_task_handoff_tool(
    *, agent_name: str, description: str | None = None
    ):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[str,"用户输入内容"],
        # these parameters are ignored by the LLM
        state: Annotated[State, InjectedState],
    ) -> Command:
        print(f"******[{agent_name}] *************接收到任务描述: {task_description}")
        message = {"role": "user", "content": task_description}
        agent_input = {"messages":state["messages"] + [message]}

        return Command(goto=[Send(agent_name, agent_input)],graph=Command.PARENT)

    return handoff_tool


handoff_chat_agent = create_task_handoff_tool(
    agent_name="chat_agent",
    description="Assign task to a chat agent.",
)

handoff_plan_agent = create_task_handoff_tool(
    agent_name="plan_agent",
    description="Assign task to a plan agent.",
)

handoff_interaction_agent = create_task_handoff_tool(
    agent_name="interaction_agent",
    description="Assign task to a interaction agent.",
)

supervisor = create_react_agent(
    model=get_llm(),
    tools=[handoff_chat_agent,
           handoff_plan_agent,
           handoff_interaction_agent,
           ],
    prompt=(
        "你是一个多智能体协调器，只负责分发任务。\n"
        "- chat_agent: 专门负责聊天类对话（如打招呼、聊天、提问回答、记忆提取等），你拥有长期的记忆，不涉及动作或物体。\n"
        "- plan_agent: 仅负责涉及“执行”、“拿起”、“放置”等具有明确动作目标的指令类任务。\n"
        "- interaction_agent: 负责机器人控制和执行一些“挥手”、“握手”等简单交互动作。\n"
        "你不应自己完成任务，只应根据对话内容选择一个工具函数（如 transfer_to_chat_agent、transfer_to_plan_agent 或 transfer_to_interaction_agent）来调用下一个 agent。\n"
        "如果对话仅是寒暄、问候、提问、记忆相关内容，应使用 transfer_to_chat_agent。\n"
        "如果对话涉及执行任务、移动物体、操作机器人，请使用 transfer_to_plan_agent。\n"
        "如果对话涉及简单的机器人交互动作（如挥手、握手等），请使用 transfer_to_interaction_agent。\n"
        "- 如果你认为任务已经完成，不需要再调用工具，请不要调用任何工具，直接停止。"
    ),
    name="supervisor"
)

loader = ConfigLoader("../config/vision_config.yaml")
yolo_cfg = loader.get_model_config("yolov8s-worldv2")

yolo_path = yolo_cfg["path"]
chat_agent = ChatAgent()
vision_agent = VisionAgent(name="vision_agent", vision_model = yolo_path)
robot_agent = RobotAgent()
plan_agent = PlanAgent(vision_agent=vision_agent, robot_agent=robot_agent)

interaction_agent = InteractionAgent(name="interaction_agent")

graph_builder = (
    StateGraph(State)
    .add_node("supervisor", supervisor)
    .add_node("chat_agent", chat_agent)
    .add_node("plan_agent", plan_agent)
    .add_node("interaction_agent", interaction_agent)

    .add_edge(START, "supervisor")
    .add_edge("chat_agent", "supervisor")
    .add_edge("interaction_agent", END)
    .add_edge("plan_agent", END)
)
runnable = graph_builder.compile(checkpointer=memory_saver)

config1 = {"configurable": {"thread_id": "session123"}}

# output1 = runnable.invoke(
#     {"messages": [{"role": "user", "content": "Hi, my name is Villala."}]},
#     config=config1
# )
# print("💬 输出:", output1["messages"][-1].content)

# state2={"messages": [{"role": "user", "content": "What is my name?"}]}

# output2 = runnable.invoke(state2,config=config1)

# print("💬 输出:", output2["messages"][-1].content)
state3 = {"messages": [{"role": "user", "content": "把篮子里的香蕉放桌上"}]}

output3 = runnable.invoke(state3,config=config1)
print("第三次输出:", output3["messages"][-1].content)

state4 = {"messages": [{"role": "user", "content": "挥手"}]}

output4 = runnable.invoke(state4,config=config1)
print("第四次输出:", output4["messages"][-1].content)


