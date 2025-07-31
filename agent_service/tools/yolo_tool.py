from ultralytics import YOLOWorld
from .config_load import ConfigLoader


class YOLO:
    def __init__(self, model_path = "/24T/yyy/wyq/llm_robot_control/agent_service/tools/weights/yolov8s-worldv2.pt", classes = None):
        # 初始化 YOLOv8-Worldv2 模型
        self.model = YOLOWorld(model_path)

        # 设置自定义标签
        self.custom_classes = classes
        self.model.set_classes(self.custom_classes)

    def __call__(self, image = '/24T/yyy/ybj/llm_robot_control/agent_service/tools/images/apple.jpg'):
        # 对指定图像执行推理，并显示结果
        results = self.model.predict(image, save=False)
        # for result in results:
        #     result.plot()
        result_dict = {}
        res = []
        for result in results:
            boxes = result.boxes  # Boxes对象，包含 xyxy、cls、conf 等信息
            for box in boxes:
                # 获取坐标（xyxy 格式）
                xyxy = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
                # 获取类别索引
                cls_id = int(box.cls.cpu().numpy()[0])
                # 获取类别名称
                class_name = self.custom_classes[cls_id]
                # 获取置信度
                conf = float(box.conf.cpu().numpy()[0])
                print(f"Detected {class_name} with confidence {conf:.2f} at bbox {xyxy}")
                r = [class_name,conf,xyxy]
                res.append(r)
        result_dict["img"]=image
        result_dict["class_name"]=class_name
        result_dict["conf"]=conf
        result_dict["box"]=xyxy
        # print("yolo result:", result_dict)
        return result_dict

if __name__ == "__main__":
    loader = ConfigLoader("../config/vision_config.yaml")
    yolo_cfg = loader.get_model_config("yolov8s-worldv2")

    yolo_path = yolo_cfg["path"]
    det = YOLO(model_path = yolo_path, classes = ['apple'])
    r = det()
    print(r)