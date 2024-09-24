import os
import json
import openai
import base64
import requests
import tempfile
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
OPENAPI_KEY = os.getenv("OPENAI_KEY")

def list_all_files_in_source_folder():
    source_folder = os.path.join(os.path.dirname(__file__), 'source')
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                yield os.path.join(root, file)

def encode_image(image_path):
    # Resize the image to 300x300
    image = Image.open(image_path)
    image = image.resize((300, 300))

    with tempfile.NamedTemporaryFile(delete=True) as temp_image:
        image.save(temp_image.name, format="JPEG")
        with open(temp_image.name, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
def classify_image(image_filename, possible_classifications):
    try:
        base64_image = encode_image(image_filename)

        # prompt = "What’s in this image?"
        prompt = "You are an expert in image classification. You are given an image and you need to classify it into one of the following categories: "
        prompt += ", ".join(possible_classifications)
        prompt = prompt + ". Please classify the image below and return only the chosen classification as a single word."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAPI_KEY}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        data = response.json()
        classification = data['choices'][0]['message']['content']

        # save the entire response alongside the file with a .json extension
        metadata_filename = image_filename + ".json"
        with open(metadata_filename, "w") as metadata_file:
            metadata_file.write(json.dumps(data, indent=4))

        return classification
    except Exception as e:
        print(f"Failed to classify image {image_filename}: {e}")
        return None



if __name__ == '__main__':
    possible_classifications = ['Portrait/People', 'Landscape/Environment', 'Still Life/Object', 'Architecture', 'Abstract/Non-representational']
    images = list_all_files_in_source_folder()
    for image in images:
        classification = classify_image(image, possible_classifications)
        filename_only = os.path.basename(image)
        print(f"{filename_only}: {classification}")
        break