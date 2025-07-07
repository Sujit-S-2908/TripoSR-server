from flask import Flask, request, jsonify, send_file
from tripoSR_inference import TripoSRModel
import os
from io import BytesIO

app = Flask(__name__)
model_path = "checkpoints/tripoSR.ckpt"

# Load model once at startup
triposr = TripoSRModel(ckpt_path=model_path)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    image_file = request.files["image"]
    output_path = triposr.run_inference(image_file)

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
