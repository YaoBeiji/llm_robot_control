import os
import yaml

class ConfigLoader:
    def __init__(self, config_path: str):
        # 解析为当前文件所在目录的相对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.abspath(os.path.join(base_dir, config_path))
        self.config_path = full_path
        self.config = self._load_yaml(full_path)

    def _load_yaml(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件未找到: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get_ollama_config(self):
        return self.config["ollama"]

    def get_model_config(self, model_name):
        for model in self.config.get("model", []):
            if model.get("name") == model_name:
                return model
        raise ValueError(f"模型未找到: {model_name}")