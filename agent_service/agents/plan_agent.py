from agent_service.config import get_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict
from agent_service.tools import extract_last_json,image_recognition_tool, normalize_instruction, parse_vision_results,camera_tool,robot_state_tool,robot_shake_hand,robot_wave_hand

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
        messages = convert_to_langchain_messages(messages)
        # my_memory = memory_dict.get(self.name, [])
        print("---------plan agent---------")

        if messages:
            print(f"[{self.name}] 接收到消息: {messages[-1].content}")
        full_messages = messages
        # print(full_messages)

        if self.use_vision_agent:
            ### use vision agent to detect objects

            extraction_prompt = """请从指令中提取目标物体 (object)、参考物体 (reference)、以及（如有明确描述）放置物体 (placement)，以JSON格式返回英文结果，例如：
                    {"object": "cup", "reference": "red box", "placement": "table"}，若有多个物体，请使用数组表示，例如：
                    {"object": ["cup", "pen"], "reference": ["box"], "placement": ["table"]}，如果某项缺失，请填入空字符串""。请注意：JSON 中所有名称需翻译为英文。指令：""" + \
                                messages[-1].content
            full_messages.append(HumanMessage(content=extraction_prompt))
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
        # print(f"[{self.name}] RobotAgent 输出: {robot_output.get('messages', [])}")
        # print(f"[{self.name}] RobotAgent memory: {robot_output.get('memory', {})}")
        return {
            "messages": robot_output.get("messages", []),
            "memory": robot_output.get("memory", {}),
        }
    