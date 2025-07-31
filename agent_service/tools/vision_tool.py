from langchain.tools import tool
import requests
import cv2
import os
import uuid
import json
import re
import ast
from .yolo_tool import YOLO

# detector = YOLO() 
TEMP_IMAGE_DIR = "./images/apple.jpg" # 测试数据地址
# os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# def extract_last_json(response):
    # 匹配所有 JSON 对象，取最后一个为所需json
    # matches = re.findall(r'\{.*?\}', response, re.DOTALL)
    # for js in reversed(matches):
    #     try:
    #         return json.loads(js)
    #     except json.JSONDecodeError:
    #         continue

    # json_str_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    # if json_str_match:
    #     json_str = json_str_match.group(1)
    #     try:
    #         data = json.loads(json_str)
    #         print("提取的JSON数据：", data)
    #         return data
    #     except json.JSONDecodeError as e:
    #         print("JSON解析错误:", e)
    # else:
    #     print("未找到JSON字符串")
    # return None

def extract_last_json(response):
    # 匹配所有 JSON 对象，取最后一个为所需 JSON
    matches = re.findall(r'\{.*?\}', response, re.DOTALL)
    for js in reversed(matches):
        try:
            data = json.loads(js)
            # print("提取的JSON数据：", data)
            return data
        except json.JSONDecodeError:
            continue
    print("未找到有效JSON字符串")
    return None

def parse_vision_results(vision_results):
    if isinstance(vision_results, dict):
        return vision_results
    elif isinstance(vision_results, str):
        # 尝试从字符串中提取大括号中的 dict 片段
        try:
            # 去掉前缀（如 "[图像识别结果]："）
            if "{" in vision_results:
                dict_str = vision_results[vision_results.index("{"):]
            else:
                return {}

            # 替换掉 `array([...], dtype=float32)` 为普通 list
            dict_str = re.sub(r'array\((\[.*?\])[^)]*\)', r'\1', dict_str)

            # 尝试解析为字典
            return ast.literal_eval(dict_str)
        except Exception as e:
            print(f"解析 vision_results 失败：{e}")
            return {}
    else:
        return {}

# # 确保将不同个数情况下的obj和ref转化为一个list
def normalize_instruction(x):
    # 如果是 None 或空字符串，返回空列表
    if x is None:
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        # 字符串按逗号分割
        return [item.strip() for item in x.split(',') if item.strip()]
    elif isinstance(x, list):
        # 如果是列表，扁平化处理，防止出现嵌套list
        flat = []
        for item in x:
            if isinstance(item, str):
                # 也可检查item是否包含逗号，拆开
                if ',' in item:
                    flat.extend([subitem.strip() for subitem in item.split(',') if subitem.strip()])
                else:
                    flat.append(item.strip())
            else:
                flat.append(item)
        return flat
    else:
        # 其他类型，强制转成字符串处理
        return [str(x)]

@tool
def image_recognition_tool(input: dict) -> str:
    """
    图像识别工具，可选择使用本地摄像头拍摄，或提供图像路径/URL。
    
    参数 input:
    - image_path_or_url: 可选，本地路径或图像URL；
    - instruction: 可选，识别命令(识别物体的特征)；
    - use_camera: 可选，布尔值，若为True则从摄像头获取图像；
    
    返回自然语言识别结果。
    """
    try:
        image_path_or_url = input.get("image_path_or_url", None)
        instruction = input.get("instruction", None)
        use_camera = input.get("use_camera", False)
        model_path = input.get("model_path", None)

        # 初始化
        detector = YOLO(model_path = model_path, classes = instruction) 

        # 摄像头拍照
        if not image_path_or_url and use_camera:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return "Error: Failed to capture image from camera."
            image_input = frame  # 直接用数组
        elif image_path_or_url:
            image_input = image_path_or_url  # 支持路径/URL
        else:
            return "Error: No image source provided (either image_path_or_url or use_camera must be set)."

        # 执行推理（支持图像路径或OpenCV图像数组）
        results = detector(image_input)
        # print("results", results)
        # img = results["img"]
        # res = results["res"]
        # if len(res) == 0:
        #     return "未检测到所需识别物体"
        # else:
        #     # 提取结果
        #     detected_objects = []
        #     for obj in res:
        #         if len(obj) == 3:
        #             name, confidence, bbox = obj
        #             if confidence>0.5:
        #                 detected_objects.append({
        #                     "class": name,
        #                     "confidence": float(confidence),
        #                     "bbox": [float(x) for x in bbox]

        #                 })
        #     # 自然语言描述
        #     descriptions = [
        #         f"{obj['class']} detected with {obj['confidence']:.2f} confidence at bbox {obj['bbox']}"
        #         for obj in detected_objects
        #     ]
        #     detected_objects.append([img])
        #     return detected_objects
        return results

    except Exception as e:
        return f"Error during image recognition: {e}"

# @tool
# def camera_tool(camera_idx):
#     """
#     相机工具，可以用来获取相机中的照片信息
    
#     参数 input:
#     - camera_idx: 必填，相机的id
    
#     返回一张图片。
#     """
#     img_idx = cv2.VideoCapture(camera_idx)
#     pass
import numpy as np
from PIL import Image
import base64
import io
from typing import List, Dict
def create_base64_color_image(color_rgb, size=(224, 224)):
    img = Image.new("RGB", size, color_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64

@tool
def camera_tool(camera_idx: List[int]) -> Dict[int, str]:
    """
    相机工具：从多个相机编号中分别拍照，并返回图片的 Base64 编码结果列表。

    参数:
    - camera_idx: 一个整型列表，每个元素是相机编号（如 [0, 1, 2]）

    返回:
    - 每个相机拍到的一张照片（Base64 编码的 JPEG 格式）组成的列表
    """
    results = {}

    # for idx in camera_idx:
    #     cap = cv2.VideoCapture(idx)
    #     if not cap.isOpened():
    #         cap.release()
    #         raise ValueError(f"无法打开相机：{idx}")

    #     ret, frame = cap.read()
    #     cap.release()

    #     if not ret:
    #         raise RuntimeError(f"无法从相机 {idx} 读取图像")

    #     # 转为 Base64
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     pil_image = Image.fromarray(frame_rgb)
    #     buffer = BytesIO()
    #     pil_image.save(buffer, format="JPEG")
    #     img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    #     results[idx] = img_base64
    results[0] =  create_base64_color_image((255, 0, 0))      # Red
    results[1] = create_base64_color_image((0, 255, 0))       # Green
    results[2] =  create_base64_color_image((0, 0, 255))      # Blue
    results[3] =  create_base64_color_image((128, 128, 128)) #

    return results
@tool
def robot_state_tool() -> List[float]:
    """
    模拟获取机器人当前状态（关节位置 + 速度），共14维。
    返回一个14维的浮点数列表。
    """
    # 用 np.ones 模拟状态，也可以改成 np.random 或实际读取
    state = np.ones((14,), dtype=float)
    return state.tolist()