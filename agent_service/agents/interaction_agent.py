from agent_service.tools import robot_wave_hand,robot_shake_hand
import requests
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

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
            result = robot_wave_hand.invoke({})
        elif "握手" in user_input or "shake" in user_input:
            result = robot_shake_hand.invoke({})
        else:
            result = {"status": "ignored", "reason": "未识别交互意图"}

        print(f"[{self.name}] 交互结果: {result}")
        return {
            "messages": [
                AIMessage(content=str(result))
            ]
        }