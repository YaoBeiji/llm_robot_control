# from openai import OpenAI
# from qwen_agent.utils.utils import get_function_description

# # Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://172.16.30.12:8090/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# plan_prompt = """你是一个机器人控制系统，负责将指令按顺序拆解给机械臂控制系统和底盘控制系统。
# You have access to the following tools:\n\n### arm_control\n\narm_control: 控制机械臂进行某些动作的服务 输入参数：[{"name": "task", "type": "string", "description": "详细描述了希望机械臂完成的任务", "required": true}] 此工具的输入应为JSON对象。\n\n### chassis_control\n\nchassis_control: 底盘控制是一个控制机器人底盘进行运动的服务。输入为运动目标点名称。返回运动结果 输入参数：[{"name": "target", "type": "string", "description": "详细描述了希望到达的目标点", "required": true}] 此工具的输入应为JSON对象。
# You will be given human language prompts, and you need to return the schemes of calling tools. Any action not in the tools must be ignored. Here are some examples.
# 指令：把桌上的可乐拿起来
# 动作：1.arm_control机械臂归零 2.chassis_control运动到书桌 3.arm_control把可口可乐拿到空中
# 指令：把桌子上的可口可乐，苹果，梨子都拿到篮子里
# 动作：1.arm_control机械臂归零 2.chassis_control运动到桌子 3.arm_control把可口可乐拿到篮子 4.arm_control把苹果拿到篮子 5.arm_control把梨子拿到篮子 6.arm_control机械臂归零
# 指令：移动到桌子旁边
# 动作：1.arm_control机械臂归零 2.chassis_control运动到桌子
# 根据例子，回答下面的问题。
# """

# class Qwen():
#     def __init__(self) -> None:
#         self.history = []
        
#     def append_history(self, role, content):
#         self.history.append({'role':role, "content":content})
        
#     def chat(self, query):
#         chat_response = client.chat.completions.create(
#             model="Qwen/Qwen2-7B-Instruct",
#             # model="Qwen1.5-32B-Chat-GPTQ-Int4",
#             messages=[
#                 {"role": "user", "content": query},
#             ]
#         )
#         print("Chat response:", chat_response)
        
        
# if __name__ == "__main__":
#     pa = Qwen()
#     query = "把桌上的水杯拿起来"
#     query = plan_prompt+"指令："+query+"动作："
#     pa.chat(query)

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.tools import BaseTool
from qwen_agent.llm import BaseChatModel
from typing import Dict, Iterator, List, Optional, Tuple, Union
from qwen_agent.llm.schema import (ASSISTANT, CONTENT, DEFAULT_SYSTEM_MESSAGE,
                                   ROLE, ContentItem, Message)
from qwen_agent.utils.utils import (get_basename_from_url,
                                    get_function_description,
                                    has_chinese_chars)
from qwen_agent.tools.robot import *
from slave_agent import *
import copy

plan_prompt = """你是一个机器人控制专家，负责将指令根据已有工具按顺序进行拆解。
You have access to the following tools:{tool_descs}
You will be given human language prompts, and you need to return the schemes of calling tools. Any action not in the tools must be ignored. Here are some examples.
{plan_demos}
根据例子，回答下面的问题。
指令：{query}
动作："""

class PlanAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict,
                                                    BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 prompt: Optional[str] = ''):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files)
        self.prompt = prompt
                
    def chat(self, messages):
        messages = copy.deepcopy(messages)
        tool_descs = '\n\n'.join(
            get_function_description(func.function)
            for func in self.function_map.values()) 
        
        if isinstance(messages[-1][CONTENT], str):
            prompt = plan_prompt.format(tool_descs=tool_descs,
                                        plan_demos=self.prompt,
                                        query=messages[-1][CONTENT])
            messages[-1][CONTENT] = prompt
        
        response = self._call_llm(messages, stream=False)
        return response
    
if __name__ == "__main__":

    llm_cfg = {
        'model': 'Qwen/Qwen2-7B-Instruct',
        # 'model': 'Qwen1.5-32B-Chat-GPTQ-Int4',
        'model_server': 'http://localhost:8090/v1',  # base_url, also known as api_base
        'api_key': 'EMPTY',
        'generate_cfg': {
            'top_p': 0.1,
            # 'temperature': 0.1,
        }
    }
    
    plan_demo = """指令：把桌上的可乐拿起来
    动作：1.arm_control机械臂归零 2.chassis_control运动到书桌 3.arm_control把可口可乐拿到空中
    指令：把桌子上的可口可乐，苹果，梨子都拿到篮子里
    动作：1.arm_control机械臂归零 2.chassis_control运动到桌子 3.arm_control把可口可乐拿到篮子 4.arm_control把苹果拿到篮子 5.arm_control把梨子拿到篮子 6.arm_control机械臂归零
    指令：移动到桌子旁边
    动作：1.arm_control机械臂归零 2.chassis_control运动到桌子"""
    tools = ['arm_control', 'chassis_control']

    pa = PlanAgent(llm=llm_cfg, function_list=tools, prompt=plan_demo)
    messages = []
    
    query = "把桌上的水杯拿起来"
    messages.append({'role': 'user', 'content': query})

    response = pa.chat(messages)
    print(response)