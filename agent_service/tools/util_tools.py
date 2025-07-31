from langchain_core.tools import tool

@tool
def object_detection(image_url: str) -> str:
    """检测图像中的所有物体，返回类别和位置信息"""
    return f"检测到红色杯子，位置在(x=128, y=240)"

@tool
def get_location() -> str:
    """返回当前SLAM定位坐标"""
    return "机器人当前位置：(1.2m, 0.8m)"

@tool
def scene_graph_builder() -> str:
    """构建当前环境的场景图"""
    return "杯子在桌子上，距离0.5米"

@tool
def plan_actions(command: str) -> str:
    """根据自然语言指令生成动作计划"""
    return "动作序列：靠近 -> 识别 -> 抓取 -> 放置"

@tool
def generate_robot_commands(actions: str) -> str:
    """将动作序列转化为底层控制命令"""
    return "已发送控制指令：MOVE, DETECT, GRASP"
