import requests
import json

# 定义服务的 URL
url = "http://localhost:3125/react"

# 定义要发送的数据
data = {
    "query": "把桌上的可乐拿起来"
}
# query = "到茶几那把水杯拿给我"
    # "query": "黑色桌子上的柠檬放到白色桌子的篮子里"


# 将数据转换为 JSON 格式
json_data = json.dumps(data)

# 发送 POST 请求并获取响应
response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

# 检查响应状态码
if response.status_code == 200:
    # 获取响应内容
    result = response.json()
    print("Response:", result)
else:
    print("Error:", response.status_code, response.text)