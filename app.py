import os
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import base64
from PIL import Image
import requests


# populate exec env
env = ""
if os.path.isfile(".env"):
    f = open(".env")
    env = f.read()
    f.close()
env = {k: v for k, v in (tuple(t.split("=")) for t in [l for l in env.split()])}


# modelName = 'tinyllama'
# modelName = 'llama3.1'
modelName = env["MODEL"] if "MODEL" in env else "llama3.1"

if modelName != "None":
    import ollama

if env["StableDiffusion"] != "None":
    from diffusers import AutoPipelineForText2Image
    import torch

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("imageGraph.html")


def generate_image(prompt):
    # Generate the image based on the prompt
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    # Save the image to a byte buffer
    buffered = BytesIO()
    image.save(buffered, format="PNG")

    # Encode the image to base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# Define the route for generating images
@app.route("/genImg", methods=["POST"])
def generate():

    user_message = request.json.get("message")
    print(f"User:{user_message}")

    if modelName != "None":
        systemPrompt = """System: You are a LLM to improve Stable Diffusion prompts from the user by adding more detail and clearly describing a scene."""
        promptContent = f"System: {systemPrompt}\n User:{user_message}"
        response = ollama.chat(model=modelName, messages=[{"role": "user", "content": promptContent}])
        imagePrompt = response["message"]["content"]
    else:
        imagePrompt = user_message

    if env["StableDiffusion"] != "None":
        generated_image = generate_image(imagePrompt)
        return jsonify({"image_base64": generated_image})
    else:
        image_url = "https://picsum.photos/1024/1024"
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")  # or 'JPEG', based on your needs
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
            return jsonify({"image_base64": img_base64})


if __name__ == "__main__":
    PORT = int(env["PORT"]) if "PORT" in env else 5002
    HOST = env["HOST"] if "HOST" in env else "127.0.0.1"
    app.run(debug=True, port=PORT, host=HOST)
