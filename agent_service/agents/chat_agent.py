from agent_service.config import get_llm

class ChatAgent:
    def __init__(self, name="chat_agent", model_name="qwen3:14b"):
        self.name = name
        self.llm = get_llm()  # use vllm qwen3

    def __call__(self, state: dict) -> dict:
        # print("🤖 chat_agent received:", state["messages"])
        response = self.llm.invoke(state["messages"])
        # print("💬 chat_agent response:", response.content)
        return {"messages": [response]}