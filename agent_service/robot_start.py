from flask import Flask, request, jsonify
from supervisor_with_description import supervisor_with_description
from modules.asr import transcribe_audio

app = Flask(__name__)


@app.route('/inference', methods=['POST'])
def inference():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio = request.files['audio']
    path = f"/tmp/{audio.filename}"
    audio.save(path)

    text = transcribe_audio(path)
    result = supervisor_with_description.invoke({"messages": [{"role": "user", "content": text}]})

    return jsonify({
        "asr_text": text,
        "final_result": result["steps"][-1]["messages"][-1].content,
        "full_steps": result["steps"]
    })