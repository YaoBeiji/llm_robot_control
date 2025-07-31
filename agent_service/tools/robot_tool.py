from langchain.tools import tool
import requests

@tool
def robot_wave_hand() -> str:
    """
    Wave the robot's hand.
    """
    try:
        response = requests.post("http://172.16.20.112:5003/wavehand",timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "action": "wave", "error": str(e)}
        
@tool
def robot_shake_hand() -> str:
    """
    Shake the robot's hand.
    """
    try:
        response = requests.post("http://172.16.20.112:5003/handshake",timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "action": "shake", "error": str(e)} 