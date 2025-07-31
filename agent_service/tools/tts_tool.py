from paddlespeech.cli.tts.infer import TTSExecutor
from playsound import playsound
import winsound


import time
tts = TTSExecutor()
s = time.time()
tts(text="今天天气十分不错。", output="./output.wav")
print("语音转换成功")
# playsound('./output.wav')
winsound.PlaySound('./output.wav', winsound.SND_FILENAME)
u = time.time()-s
print(u)