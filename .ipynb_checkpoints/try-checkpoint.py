import requests
import json

# 定义请求的URL
url = 'http://192.168.0.68:5003/handshake'

# 准备请求数据
data = {}

# 发送POST请求
response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

# 打印服务器响应
if response.status_code == 200:
    print("Request was successful!")
    print("Response data:", response.json())
else:
    print("Request failed!")
    print("Status code:", response.status_code)
    print("Response text:", response.text)