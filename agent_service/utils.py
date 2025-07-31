from dotenv import load_dotenv
import os
from qwen_agent.agents import ReActChat
from qwen_agent.agents import PlanAgent
from qwen_agent.tools.base import BaseTool, register_tool, TOOL_REGISTRY
import yaml
from typing import Union
import json
from custom_log import LOG
from qwen_agent.tools.robot import *


load_dotenv()

def clear_screen():
    # 判断操作系统类型，执行相应的清屏命令
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix/Linux/macOS
        os.system('clear')

def init_agent_service(isLocal, atype, tools, prompt):

    if isLocal:
        llm_cfg = {
            'model': os.getenv("LOCAL_QWEN_MODEL"),
            'model_server': os.getenv("LOCAL_SERVER"),  # base_url, also known as api_base
            'api_key': os.getenv("LOCAL_QWEN_API_KEY"),
            'generate_cfg': {
                'top_p': 0.1,
            }
        }
    else:
        llm_cfg = {
            'model': os.getenv("CLOUD_QWEN_MODEL"),
            'model_server': os.getenv("CLOUD_SERVER"),
            'api_key': os.getenv("CLOUD_QWEN_API_KEY"),
            'generate_cfg': {
                'top_p': 0.1,
            }
        }
    bot = None
    if atype == 'plan':
        bot = PlanAgent(llm=llm_cfg, function_list=tools, prompt=prompt)
    elif atype == 'react':
        bot = ReActChat(llm=llm_cfg, function_list=tools, prompt=prompt)
    return bot


def create_slave_agent_class(agent_config):
    class_name = f"{agent_config['name']}_SlaveAgent"
    
    class SlaveAgentClass(BaseTool):
        description = agent_config['description']
        parameters = agent_config['parameters']
        tools = agent_config['tools']
        react_prompt = agent_config['react_prompt']
        plan_prompt = agent_config['plan_prompt']
        is_local = agent_config['isLocal']

        def call(self, params: Union[str, dict], **kwargs) -> str:
            params = self._verify_json_format_args(params)
            task = params['task']
            
            messages = []
            messages.append({'role': 'user', 'content': task})
            
            plan_bot = init_agent_service(self.is_local, 'plan', self.tools, self.plan_prompt)
            sug = plan_bot.chat(messages)
            demo = sug[0].content.replace('\n','')
            LOG.info(demo)

            prompt = self.react_prompt.format(demo=demo)
            react_bot = init_agent_service(self.is_local, 'react', self.tools, prompt)
            # react_bot = init_react_service(self.tools, prompt)
            
            last_response = None
            for response in react_bot.run(messages):
                clear_screen()
                print(f'{self.__class__.__name__} response: {response}')
                last_response = response
            LOG.info(last_response)
            result = last_response[0]['content'].split("Final Answer:")[1].strip()
            return json.dumps({"result": result}, ensure_ascii=False)
            # return json.dumps({"result": "success"}, ensure_ascii=False)

    # 使用type动态创建类,并赋予唯一的类名
    UniqueSlaveAgentClass = type(class_name, (SlaveAgentClass,), {})
    register_tool(agent_config['name'])(UniqueSlaveAgentClass)
    return UniqueSlaveAgentClass


def create_slave_agents(config):
    slave_agents = {}
    for agent_config in config['slave_agents']:
        AgentClass = create_slave_agent_class(agent_config)
        slave_agents[agent_config['name']] = AgentClass()
    return slave_agents
