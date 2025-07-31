import time
import json
from typing import Union
from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('detect_anything')
class DetectAnything(BaseTool):
    description = '万物检测是一个2D图像目标检测服务。输入物体文本描述，返回物体在图像中的坐标'
    parameters = [{
        'name': 'target',
        'type': 'string',
        'description': '详细描述了希望检测的目标是什么',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        import random
        # 生成 4 个随机整数
        random_list = [random.randint(1, 100) for _ in range(4)]    
        return json.dumps({'label':params['target'], 'box':random_list}, ensure_ascii=False)

@register_tool('arm_point_to_point_move')
class ArmMoveTo(BaseTool):
    description = '机械臂点到点动作是一个控制机械臂运动到任意位置的服务。输入目标位置，返回机械臂运动结果。'
    parameters = [{
        'name': 'box',
        'type': 'list',
        'description': '物品的最小包围矩形，包含4个元素，',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        return json.dumps({"result":"success"}, ensure_ascii=False)
    
@register_tool('arm_lift')
class ArmLift(BaseTool):
    description = '机械臂抬起是一个控制机械臂升到空中的服务。输入为空，返回机械臂运动结果。'
    parameters = []

    def call(self, params: Union[str, dict], **kwargs) -> str:
        return json.dumps({"result":"success"}, ensure_ascii=False)


@register_tool('grasp')
class Grasp(BaseTool):
    description = '夹取动作是一个控制机械臂夹爪抓取物体的服务。输入为空。返回夹取运动结果。'
    parameters = []
    def call(self, params: Union[str, dict], **kwargs) -> str:
        return json.dumps({"result":"success"}, ensure_ascii=False)

@register_tool('release')
class Release(BaseTool):
    description = '释放动作是一个控制机械臂夹爪释放物体的服务。输入为空。返回释放运动结果。'
    parameters = []
    def call(self, params: Union[str, dict], **kwargs) -> str:
        return json.dumps({"result":"success"}, ensure_ascii=False)
    
@register_tool('arm_reset')
class ArmRest(BaseTool):
    description = '机械臂归零是一个控制机械臂进行归零服务。输入为空。返回运动结果'
    parameters = []

    def call(self, params: Union[str, dict], **kwargs) -> str:
        return json.dumps({"result":"success"}, ensure_ascii=False)
    
@register_tool('chassis_point_to_point_move')
class ChassisMoveTo(BaseTool):
    description = '底盘控制是一个控制机器人底盘进行运动的服务。输入为运动目标点名称。返回运动结果'
    parameters = [{
        'name': 'target',
        'type': 'string',
        'description': '详细描述了希望到达的目标点',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        return json.dumps({"result":"success"}, ensure_ascii=False)