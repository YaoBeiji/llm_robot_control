import yaml
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def load_config(path="agent_service/config/szx.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
def get_llm(tools=None):
    config = load_config()
    provider = config.get("llm_provider")
    if provider == "ollama":
        ollama_cfg = config["ollama"]
        llm = ChatOllama(
            model=ollama_cfg["model"],
            base_url=ollama_cfg["base_url"],
            temperature=ollama_cfg.get("temperature", 0.7)
        )
    elif provider == "vllm":
        vllm_cfg = config["vllm"]
        print(vllm_cfg["model"],vllm_cfg["openai_api_base"])
        if tools is None:
            llm = ChatOpenAI(
                model=vllm_cfg["model"],
                base_url=vllm_cfg["openai_api_base"],
                api_key=vllm_cfg["openai_api_key"],
                temperature=vllm_cfg.get("temperature", 0.7),
            )
        else:
            llm = ChatOpenAI(
                model=vllm_cfg["model"],
                base_url=vllm_cfg["openai_api_base"],
                api_key=vllm_cfg["openai_api_key"],
                temperature=vllm_cfg.get("temperature", 0.7),
                model_kwargs={
                    "tool_choice": "auto" , # 👈 关键设置
                    "tools": [tool.dict() for tool in tools],
                }
            )
    else:
        raise ValueError(f"Unsupported llm_provider: {provider}")
    return llm