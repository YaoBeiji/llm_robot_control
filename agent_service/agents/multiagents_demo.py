from langgraph.types import Send
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models import ChatOllama
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from agent_service.tools.config_load import ConfigLoader
from agent_service.memory import MultiAgentState
from langchain_openai import ChatOpenAI
from agent_service.tools import extract_last_json,image_recognition_tool, normalize_instruction, parse_vision_results,camera_tool,robot_state_tool,robot_shake_hand,robot_wave_hand
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START,END, MessagesState
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import asyncio
import json
from langgraph.graph.message import add_messages
from agent_service.config import get_llm
import ast
import base64
from typing import List, Dict
from PIL import Image
import io
import numpy as np
import requests
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]

memory_saver = MemorySaver()

def base64_to_model_image(base64_str: str, target_size=(224, 224)) -> np.ndarray:
    # 解码 base64 为字节流
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 调整大小为目标模型输入尺寸
    img = img.resize(target_size)

    # 转为 numpy 数组，格式为 (H, W, C)
    img_np = np.array(img, dtype=np.uint8)

    # 转换为模型要求的格式 (3, 224, 224)
    img_np = np.transpose(img_np, (2, 0, 1))  # 从 (H, W, C) -> (C, H, W)

    return img_np

def safe_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj]
    else:
        return obj
# 示例：将 dict 转为 LangChain 消息对象
def convert_to_langchain_messages(messages):
    converted = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                # 可选：处理 system 等其他角色
                converted.append(SystemMessage(content=content))
        else:
            converted.append(msg)
    return converted


class VisionAgent:
    def __init__(self, name="vision_agent", vision_model = None):
        self.name = name
        self.llm = get_llm() #use vllm qwen3
        self.model = vision_model

    # origin_messages:原本的chat agent输入 state:chat agent输出
    def __call__(self,origin_messages:str,  state: dict) -> dict:
        messages = state.get("messages", [])

        # 获取传入消息
        if not messages:
            response = HumanMessage(content="未检测到用户指令。")
            print(f"[{self.name}] 没有接收到消息")
            return {"messages": messages + [response]}
        print(f"[{self.name}] 接收到消息: {messages}")
        response = messages[-1].content

        # 将response解析为json格式
        result = extract_last_json(response)
        if result is None:
            print("无法解析 JSON 结果！")
            print("无法解析 JSON，模型返回内容为：", messages)
            # print("错误信息：", e)
            obj = None
            ref = None
            pla = None
        else:
            print("json解析结果:", result)
            obj = result.get("object", "未知")
            ref = result.get("reference", "未知")
            pla = result.get("placement", "未知")
            print("目标物体：", obj)
            print("参考物体：", ref)
            print("放置目标：", pla)

        # 利用关键词调用vision tool检测

        """
        利用LLM提取得到的物体和关键词采用vision tool进行物体检测
        """

        if obj:
            print("[vision agent]目标：",
                  normalize_instruction(obj) + normalize_instruction(ref) + normalize_instruction(pla))
            result = image_recognition_tool.invoke({
                "input": {
                    "image_path_or_url": r"/24T/yyy/wyq/llm_robot_control/agent_service/tools/images/1.jpg",
                    # 是否采用本地数据
                    "use_camera": False,  # 是否采用相机实时输入
                    "instruction": normalize_instruction(obj) + normalize_instruction(ref) + normalize_instruction(pla),
                    "model_path": self.model
                }
            })
            if type(result) != str:
                result["message"] = origin_messages
                print(f"[图像识别结果]：{result}")
                response = AIMessage(content=f"[图像识别结果]：{result}")
            else:
                print("未识别到图像中需要检测的物体。")
                response = AIMessage(content="[图像识别结果]：未识别到图像中需要检测的物体。")
        else:
            # 正常调用模型
            print("未识别到对话中需要检测的物体。")
            response = AIMessage(content="[图像识别结果]：未识别到图像中需要检测的物体。")

        return response

class RobotAgent:
    def __init__(self, name="robot_agent", model_name="qwen:14b"):
        self.name = name

    def __call__(self, state: dict = None,use_vision_agent:bool = False, camera_idx: List[int] = [0,1,2,3]) -> dict:
        messages = state.get("messages", [])
        memory_dict = state.get("memory", {})
        my_memory = memory_dict.get(self.name, [])
        self.use_vision_agent = use_vision_agent
        self.camera_idx = camera_idx
        if self.use_vision_agent:
            vision_results = state.get("detected_positions", {})  # vision_agent 提供
            vision_results = parse_vision_results(vision_results)
            img = vision_results["img"]
            img = None
            # 调用 VLA 辅助意图解析（可选，但推荐）
            vla_response = self.query_vla(img, messages[-1].content if messages else "")

            vla_response = HumanMessage(content=json.dumps(vla_response, ensure_ascii=False))
            new_memory_dict = {
                **memory_dict,
                # self.name: my_memory + messages + [llm_response]
                self.name: my_memory + messages + [vla_response]
            }
            return {
                "messages": messages + [vla_response],
                "memory": new_memory_dict,
                # "robot_action": parsed_response
            }

        else:
            img = state.get("image", None)
            if img is not None:
                for camera_idx in self.camera_idx:
                    if camera_idx in img: 
                        # camera_idx 0 -->cam_high ,1-->cam_low , 2--> cam_left_wrist, 3-->cam_right_wrist
                        if camera_idx == 0:
                            img_cam_high = img[camera_idx]
                        elif camera_idx == 1:
                            img_cam_low = img[camera_idx]
                        elif camera_idx == 2:
                            img_cam_left_wrist = img[camera_idx]
                        elif camera_idx == 3:
                            img_cam_right_wrist = img[camera_idx]
                state = state.get("state", None)
                if state is not None:
                    vla_response = self.query_vla_pi0(img_cam_high,img_cam_low,img_cam_left_wrist, img_cam_right_wrist , messages[-1].content if messages else "",state)
                    print(f"[RobotAgent] VLA 响应: {vla_response}")
                    vla_response = AIMessage(content="VLA执行完毕，已生成动作序列")

                    new_memory_dict = {
                        **memory_dict,
                        self.name: my_memory + messages + [vla_response]
                    }
                    return {
                        "messages": [vla_response],
                        "memory": new_memory_dict
                    }
                else:
                    raise ValueError("未提供机器人状态，请检查Robot输入状态。")
            else:
                raise ValueError("未提供图像数据，请检查Robot输入状态。")

    def query_vla_pi0(self, img_cam_high, img_cam_low, img_cam_left_wrist, img_cam_right_wrist, prompt: str,state:None) -> str:
        """
        调用 VLA 模块进行视觉语言分析，辅助意图推理。
        返回结果应该是对任务更详细的理解或解析信息。
        """
        from openpi.training import config
        from openpi.policies import policy_config
        from openpi.shared import download
        import numpy as np
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        os.environ["OPENPI_DATA_HOME"] = "/24T/yyy/szx/openpi/download_model/openpi"
        config = config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
        if state is None:
            state = np.ones((14,), dtype=np.float32)

        inputvla ={
            "state": state,
            "images": {
                "cam_high": base64_to_model_image(img_cam_high),
                "cam_low": base64_to_model_image(img_cam_low),
                "cam_left_wrist": base64_to_model_image(img_cam_left_wrist),
                "cam_right_wrist": base64_to_model_image(img_cam_right_wrist),
            },
            "prompt": prompt,
        }
        try:
            action_chunk = policy.infer(inputvla)["actions"]
            print(action_chunk.shape)
            action_chunk = {"action": action_chunk}
            action_chunk =safe_json(action_chunk)  # 确保返回的 JSON 可序列化
            return action_chunk
        except Exception as e:
            return f"VLA调用异常: {str(e)}"

    def _build_prompt(self, messages, vision_results, vla_result):
        task = messages[-1].content if messages else ""
        instruction = (
            "你是一个机器人控制agent。请根据用户的任务描述、视觉模块提供的物体位置、"
            "以及VLA模块提供的推理结果，输出如下格式的 JSON 控制指令：\n\n"
            '{\n'
            '  "action": "pick/place",\n'
            '  "object": "<物体名>",\n'
            '  "target_location": [x, y, z]\n'
            '}\n\n'
        )
        vision_info = f"视觉识别结果（vision_agent）:\n{json.dumps(vision_results, ensure_ascii=False)}"
        vla_info = f"\nVLA推理结果:\n{vla_result}"
        return instruction + f"\n用户任务:\n{task}\n\n" + vision_info + vla_info

    def query_vla(self, img, user_instruction: str) -> str:
        """
        调用 VLA 模块进行视觉语言分析，辅助意图推理。
        返回结果应该是对任务更详细的理解或解析信息。
        """
        import requests

        try:
            if img is not None:
                resp = requests.post(
                    self.vla_url,
                    json={"instruction": user_instruction}
                )
                if resp.status_code == 200:
                    return resp.json().get("analysis", "无推理结果")
                else:
                    return f"VLA错误: HTTP {resp.status_code}"
            else:
                resp = {"desk":[1,1,1], "basket":[1,1.1,1.1],"banana":[1,1.1,1.2]}
                return resp

        except Exception as e:
            return f"VLA调用异常: {str(e)}"


class PlanAgent:
    def __init__(self, name="plan_agent", vision_agent=None, robot_agent=None,use_vision_agent:bool = False, camera_idx: List[int] = [0,1,2,3]):
        self.llm = get_llm() #use vllm qwen3
        self.name = name
        self.vision_agent = vision_agent
        self.robot_agent = robot_agent
        self.use_vision_agent = use_vision_agent
        self.camera_idx = camera_idx

    def __call__(self, state: dict) -> dict:
        messages = state.get("messages", [])
        memory_dict = state.get("memory", {})
        messages = convert_to_langchain_messages(messages)
        my_memory = memory_dict.get(self.name, [])
        print("---------plan agent---------")
        if messages:
            print(f"[{self.name}] 接收到消息: {messages[-1].content}")
        full_messages = my_memory + messages 
        #[HumanMessage(content='将篮子里的香蕉拿起并放置到桌面上。', additional_kwargs={}, response_metadata={})]
        print(full_messages)

        if self.use_vision_agent:
            ### use vision agent to detect objects

            extraction_prompt = """请从指令中提取目标物体 (object)、参考物体 (reference)、以及（如有明确描述）放置物体 (placement)，以JSON格式返回英文结果，例如：
                    {"object": "cup", "reference": "red box", "placement": "table"}，若有多个物体，请使用数组表示，例如：
                    {"object": ["cup", "pen"], "reference": ["box"], "placement": ["table"]}，如果某项缺失，请填入空字符串""。请注意：JSON 中所有名称需翻译为英文。指令：""" + \
                                messages[-1].content
            full_messages.append(HumanMessage(content=extraction_prompt))
            print(f"--full_messages--: {full_messages}")
            response = self.llm.invoke(full_messages)
            print(f"[chat_agent] 解析任务指令为: {response.content}")

            # 先调用VisionAgent检测物体
            vision_state = {"messages": [HumanMessage(content=response.content)], "memory": state.get("memory", {})}
            vision_output = self.vision_agent(messages[-1].content, vision_state)
            detected_positions = vision_output.content
            robot_messages = [HumanMessage(content=response.content)]
            img_dict ={}

        else:
            img_dict = camera_tool.invoke({"camera_idx":self.camera_idx})
            state =robot_state_tool.invoke({})
            detected_positions=None
            robot_messages =full_messages

        # 把检测结果放入状态，调用RobotAgent执行
        robot_state = {
            "messages": robot_messages,
            # "memory": vision_output.get("memory", {}),
            "state" : state,
            "detected_positions": detected_positions,
            "image":img_dict
        }
        robot_output = self.robot_agent(robot_state,self.use_vision_agent,self.camera_idx)

        # # 返回机器人执行结果，更新memory
        print(f"[{self.name}] RobotAgent 输出: {robot_output.get('messages', [])}")
        # print(f"[{self.name}] RobotAgent memory: {robot_output.get('memory', {})}")
        return {
            "messages": robot_output.get("messages", []),
            "memory": robot_output.get("memory", {}),
        }

class ChatAgent2:
    def __init__(self, name="chat_agent", model_name="qwen3:14b"):
        self.name = name
        self.llm = get_llm()  # use vllm qwen3

    def __call__(self, state: dict) -> dict:
        # print("🤖 chat_agent received:", state["messages"])
        response = self.llm.invoke(state["messages"])
        # print("💬 chat_agent response:", response.content)
        return {"messages": [response]}

class InteractionAgent:
    def __init__(self, name="interaction_agent"):
        self.name = name

    def __call__(self, state: dict) -> dict:
        messages = state.get("messages", [])
        user_input = ""

        if isinstance(messages, list):
            # 拼接所有用户消息内容
            user_input = " ".join(m.content if hasattr(m, "content") else m.get("content", "") for m in messages)

        print(f"[{self.name}] 接收到任务描述: {user_input}")
        
        # 简单意图识别：你可以替换为更复杂的模型或关键词字典
        if "挥手" in user_input or "wave" in user_input:
            result = self._wave()
        elif "握手" in user_input or "shake" in user_input:
            result = self._shake()
        else:
            result = {"status": "ignored", "reason": "未识别交互意图"}

        print(f"[{self.name}] 交互结果: {result}")
        return {
            "messages": [
                AIMessage(content=str(result))
            ]
        }

    def _wave(self):
        try:
            response = requests.post("http://172.16.20.112:5003/wavehand",timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                        "messages": [
                            AIMessage(content="挥手成功，已完成！"),
                            {"role": "system", "content": "任务已完成，不需要进一步操作。"}
                        ]
                    }
            return {"status": "error", "action": "wave", "error": str(e)}

    def _shake(self):
        try:
            response = requests.post("http://172.16.20.112:5003/handshake",timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "action": "shake", "error": str(e)}  
            

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
        # agent_input = {**state, "messages": [message]}
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
           # handoff_robot_agent
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
        "如果任务已经完成，不需要再调用工具，请不要调用任何工具，直接停止。"
    ),
    # prompt=(
    #     "你是一个任务调度员。\n"
    #     "根据用户输入判断该交给哪个智能体来处理。\n"
    #     "- 如果是日常对话、聊天、记忆类内容，请调用 transfer_to_chat_agent 工具。\n"
    #     # "- 如果你认为任务已经完成，不需要再调用工具，请不要调用任何工具，直接停止。\n"
    #     "你不应该亲自回答，只负责分发任务。"
        
    # ),
    name="supervisor"
)

loader = ConfigLoader("../config/vision_config.yaml")
yolo_cfg = loader.get_model_config("yolov8s-worldv2")

yolo_path = yolo_cfg["path"]
chat_agent = ChatAgent2()
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
    # .add_edge("plan_agent", "supervisor")
    .add_edge("interaction_agent", "supervisor")
    .add_edge("plan_agent", END)
    # .add_edge("supervisor", END)
    # .compile(checkpointer=memory_saver)
)
runnable = graph_builder.compile(checkpointer=memory_saver)


# if __name__ == "__main__":
config1 = {"configurable": {"thread_id": "session123"}}

output1 = runnable.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is Villala."}]},
    config=config1
)
print("💬 输出:", output1["messages"][-1].content)

state2={"messages": [{"role": "user", "content": "What is my name?"}]}

output2 = runnable.invoke(state2,config=config1)

print("💬 输出:", output2["messages"][-1].content)
state4 = {"messages": [{"role": "user", "content": "挥手"}]}

output4 = runnable.invoke(state4,config=config1)
print("第四次输出:", output4["messages"][-1].content)
state3 = {"messages": [{"role": "user", "content": "把篮子里的香蕉放桌上"}]}

output3 = runnable.invoke(state3,config=config1)
print("第三次输出:", output3["messages"][-1].content)
