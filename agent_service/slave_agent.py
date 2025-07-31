from qwen_agent.agents import ReActChat
from qwen_agent.agents import PlanAgent

from qwen_agent.tools.robot import *
import time
import json
from typing import Union
from qwen_agent.tools.base import BaseTool, register_tool
from custom_log import LOG
import os

API_KEY = "sk-bc1ed607a44f45e89d436deb4ea80d2d"
QWEN_MODEL = "qwen-max"

def clear_screen():
    # 判断操作系统类型，执行相应的清屏命令
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix/Linux/macOS
        os.system('clear')

def init_plan_service(tools, prompt="",  isLocal=True):

    if isLocal:
        llm_cfg = {
            'model': 'Qwen/Qwen2-7B-Instruct',
            # 'model': 'Qwen1.5-32B-Chat-GPTQ-Int4',
            'model_server': 'http://172.16.30.12:8090/v1',  # base_url, also known as api_base
            'api_key': 'EMPTY',
            'generate_cfg': {
                'top_p': 0.1,
                # 'temperature': 0.1,
            }
        }
    else:
        llm_cfg = {
            'model': QWEN_MODEL,
            'model_server': 'dashscope',
            'api_key': API_KEY,
            'generate_cfg': {
                'top_p': 0.1,
                # 'temperature': 0.1,
            }
        }

    bot = PlanAgent(llm=llm_cfg, function_list=tools, prompt=prompt)
    return bot
        

def init_react_service(tools, prompt="",  isLocal=True):

    if isLocal:
        llm_cfg = {
            'model': 'Qwen/Qwen2-7B-Instruct',
            # 'model': 'Qwen1.5-32B-Chat-GPTQ-Int4',
            'model_server': 'http://172.16.30.12:8090/v1',  # base_url, also known as api_base
            'api_key': 'EMPTY',
            'generate_cfg': {
                'top_p': 0.1,
                'temperature': 0.1,
            }
        }
    else:
        llm_cfg = {
            'model': QWEN_MODEL,
            'model_server': 'dashscope',
            'api_key': API_KEY,
            'generate_cfg': {
                'top_p': 0.1,
                'temperature': 0.1,
            }
        }

    bot = ReActChat(llm=llm_cfg, function_list=tools, prompt=prompt)
    return bot

@register_tool('arm_control')
class ArmSystem(BaseTool):
    description = '控制机械臂进行某些动作的服务'
    parameters = [{
        'name': 'task',
        'type': 'string',
        'description': '详细描述了希望机械臂完成的任务',
        'required': True
    }]
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        

        tools = ["detect_anything","arm_point_to_point_move",'grasp', "release", "arm_reset", "arm_lift"]

        agent_prompt = """你必须按照如下思路解决这个问题：
{demo}"""

        plan_demo = """
    指令：机械臂归零
    动作：1.arm_reset机械臂归零
    指令：把水果拿到篮子里
    动作：1.detect_anything检测水果的位置 2.arm_point_to_point_move机械臂移动到水果 3.grasp抓取水果  4.detect_anything检测到篮子的位置 5.arm_point_to_point_move机械臂移动到到篮子上 6.release水果
    指令：可乐拿到空中
    动作：1.detect_anything检测可乐的位置 2.arm_point_to_point_move机械臂移动到可乐 3.grasp抓取可乐 4.arm_lift抬起机械臂
    """        

        
        params = self._verify_json_format_args(params)
        task = params['task']
        
        messages = []
        messages.append({'role': 'user', 'content': task})
        
        plan_bot = init_plan_service(tools, plan_demo)
        sug = plan_bot.chat(messages)
        demo = sug[0].content
        LOG.info(demo)
        prompt = agent_prompt.format(demo=demo)
        
        react_bot = init_react_service(tools, prompt)
        
        last_response = None
        for response in react_bot.run(messages):
            last_response = response
        LOG.info(last_response)
        result = last_response[0]['content'].split("Final Answer:")[1].strip()
        return json.dumps({"result":result}, ensure_ascii=False)
        
@register_tool('chassis_control')
class ChassisSystem(BaseTool):
    description = '控制底盘进行某些动作的服务'
    parameters = [{
        'name': 'task',
        'type': 'string',
        'description': '详细描述了希望底盘完成的任务',
        'required': True
    }]
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        task = params['task']
        
        messages = []
        messages.append({'role': 'user', 'content': task})
        
        tools = ["chassis_point_to_point_move"]

        agent_prompt = """你必须按照如下思路解决这个问题：
{demo}"""

        plan_demo = """
    指令：运动到桌子
    动作：1.chassis_point_to_point_move桌子
    """
        
        plan_bot = init_plan_service(tools, plan_demo)
        sug = plan_bot.chat(messages)
        demo = sug[0].content
        LOG.info(demo)
        prompt = agent_prompt.format(demo=demo)
        
        react_bot = init_react_service(tools, prompt)
        
        last_response = None
        for response in react_bot.run(messages):
            last_response = response
        LOG.info(last_response)
        result = last_response[0]['content'].split("Final Answer:")[1].strip()
        return json.dumps({"result":result}, ensure_ascii=False)
        # return json.dumps({"result":"success"}, ensure_ascii=False)
        