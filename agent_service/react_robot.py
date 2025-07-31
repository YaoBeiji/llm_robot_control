from flask import Flask, jsonify, request
from qwen_agent.tools.robot import *
from utils import init_agent_service, create_slave_agents, clear_screen
import yaml
import time
from custom_log import LOG
from qwen_agent.tools.base import TOOL_REGISTRY

app = Flask(__name__)

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
    
# 先注册tools，再注册slave，最后注册master
slave_agents = create_slave_agents(config)

@app.route("/react", methods=["POST"])
def run_react_robot():
    messages = []
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Missing parameters"}), 400
    
    messages = [{'role': 'user', 'content': data['query']}]
    # 访问master_agent的信息
    master_agent = config['master_agent']
    plan_bot = init_agent_service(master_agent['isLocal'], 'plan', master_agent['tools'], master_agent['plan_prompt'])
    sug = plan_bot.chat(messages)
    demo = sug[0].content.replace('\n','')
    LOG.info(demo)
    
    prompt = master_agent['react_prompt'].format(demo=demo)
    react_bot = init_agent_service(master_agent['isLocal'], 'react', master_agent['tools'], prompt)
    
    last_response = None
    for response in react_bot.run(messages):
        clear_screen()
        print('master response:', response)
        last_response = response
    LOG.info(last_response)
    return jsonify({"result": "Agent Finished", "last_response": last_response})

if __name__ == "__main__":
    app.run(port=3125, debug=True)