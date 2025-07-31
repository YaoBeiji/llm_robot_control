from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, MessagesState
import json
import requests
from langchain_core.messages import HumanMessage,AIMessage
from typing import List, Dict
from agent_service.tools import extract_last_json,image_recognition_tool, normalize_instruction, parse_vision_results,camera_tool,robot_state_tool,robot_shake_hand,robot_wave_hand
import numpy as np
import base64
from PIL import Image
import io

def safe_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj]
    else:
        return obj

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
                    # print(f"[RobotAgent] VLA 响应: {vla_response}")
                    # vla_response = HumanMessage(content=json.dumps(vla_response, ensure_ascii=False))
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

