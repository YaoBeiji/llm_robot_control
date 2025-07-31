from langchain_core.messages import HumanMessage
from typing import Dict, Any
from langchain_community.chat_models import ChatOllama
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from agent_service.tools.vision_tool import image_recognition_tool, extract_last_json, normalize_instruction
from agent_service.tools.config_load import ConfigLoader
from .chat_agent import ChatAgent
import json
import re

class VisionAgent:
    def __init__(self, name="vision_agent", vision_model = None):
        self.name = name
        self.model = vision_model

    # origin_messages:原本的chat agent输入 state:chat agent输出
    def __call__(self, origin_messages:str, state: dict) -> dict:
        messages = state.get("messages", [])

        # 获取传入消息
        if not messages:
            response = HumanMessage(content="未检测到用户指令。")
            print(f"[{self.name}] 没有接收到消息")
            return {"messages": messages + [response]}
        
        response = messages[-1].content
        print(response)
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
            print("vision agent目标：", normalize_instruction(obj) + normalize_instruction(ref) + normalize_instruction(pla))
            result = image_recognition_tool.invoke({
            "input": {
                "image_path_or_url": r"/24T/yyy/wyq/llm_robot_control/agent_service/tools/images/apple.jpg", # 是否采用本地数据
                "use_camera": False, # 是否采用相机实时输入
                "instruction": normalize_instruction(obj) + normalize_instruction(ref) + normalize_instruction(pla),
                "model_path": self.model
            }
        })
            if type(result) != str:
                print(f"[图像识别结果]：{result}")
                result["message"] = origin_messages
            else:
                print("未识别到图像中需要检测的物体。")
                response = AIMessage(content="[图像识别结果]：未识别到图像中需要检测的物体。")
        else:
            # 正常调用模型
            print("未识别到对话中需要检测的物体。")
            response = AIMessage(content="[图像识别结果]：未识别到图像中需要检测的物体。")

        return response


if __name__ == "__main__":

    loader = ConfigLoader("../config/vision_config.yaml")
    yolo_cfg = loader.get_model_config("yolov8s-worldv2")

    yolo_path = yolo_cfg["path"]

    config = loader.config
    ollama_cfg = config["ollama"]

    chat_agent = ChatAgent(name="chat_agent", model_config = ollama_cfg)
    vision_agent = VisionAgent(name="vision_agent", vision_model = yolo_path)

    message = "帮我拿一个苹果"

    # 模拟状态传入
    state = {
        "messages": [HumanMessage(content=message)],
        "memory": {}
    }

#
# 指令：把桌上的可乐拿起来
#     动作：1.arm_control机械臂归零 2.chassis_control运动到书桌 3.arm_control把可口可乐拿到空中
#     指令：把桌子上的可口可乐，苹果，梨子都拿到篮子里
#     动作：1.arm_control机械臂归零 2.chassis_control运动到桌子 3.arm_control把可口可乐拿到篮子 4.arm_control把苹果拿到篮子 5.arm_control把梨子拿到篮子 6.arm_control机械臂归零
#     指令：移动到桌子旁边
#     动作：1.arm_control机械臂归零 2.chassis_control运动到桌子
#

    response = chat_agent(state)
    print("\n[ChatAgent 输出]:", response["messages"][-1].content)

    obj_state = vision_agent(message, response)
    print(f"[图像识别结果]：{obj_state}")
