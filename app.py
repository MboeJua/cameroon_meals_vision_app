import gradio as gr
import os
import io
import json
import torch
from google.cloud import storage
from fastai.vision.all import *
from fastai.learner import Learner
from pathlib import Path
torch.serialization.add_safe_globals([Learner])

#Setting up GCP client
credentials_content = os.environ['gcp_cam']
with open('gcp_key.json', 'w') as f:
    f.write(credentials_content)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_key.json'
bucket_name = os.environ['gcp_bucket']
pkl_blob = 'paulinus/cameroon_food_weights.pth'
dataset_blob = 'paulinus/cameroon_meals/dataset.zip'
local_pkl = Path('cameroon_food_weight.pth')
local_zip = 'dataset.zip'
local_dataset_path = Path('dataset')


client = storage.Client()
bucket = client.bucket(bucket_name)

#Weights
blob = bucket.blob(pkl_blob)
blob.download_to_filename(local_pkl)

#Dataset
bucket.blob(dataset_blob).download_to_filename(local_zip)

# Extract dataset.zip
with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall('dataset')


#NN
dls = ImageDataLoaders.from_folder(
    local_dataset_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(mult=1.0)
)



learn = vision_learner(dls, resnet34, metrics=accuracy)
load_model(local_pkl, learn.model, learn.opt)



def resize_image(img_path, max_width=640, max_height=480):
    img = Image.open(img_path)
    img.thumbnail((max_width, max_height))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    buf.seek(0)
    return buf

def predict(img):
    resized_img = resize_image(img)
    pred_class, pred_idx, outputs = learn.predict(PILImage.create(resized_img))
    prob = outputs[pred_idx].item()
    return f"Meal: {pred_class}, Probability: {prob:.4f}"

#Build Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", sources=["upload", "webcam","clipboard"]),
    outputs=gr.Textbox(),
    title="Cameroonian Meal Recognizer",
    description="""<h2>Discover Authentic Cameroonian Meals!</h2>
                   <p><b>Welcome to the Cameroonian Meal Recognizer (Version 1):</b> An AI tool designed to help you identify traditional Cameroonian dishes from a photo.</p>
                   <p><mark>Whether you're a food lover or just exploring Cameroon's rich cuisines, this tool offers a friendly playground to learn about our diverse dishes.</mark></p>

                   <p>Future updates will add features like:
                   <ul>
                     <li>Ingredient lists</li>
                     <li>Meal preparation details</li>
                     <li>Origin (locality) information</li>
                     <li>Nearby restaurants</li>
                   </ul>
                   </p>
                   <p><i>Upload a photo of a meal, and our AI will identify it, providing you with the predicted dish name and probability score.</i></p>
                   <p><u>Perfect for food lovers, chefs, or anyone looking to explore the unique and diverse flavors of Cameroon.</u></p>
                   <p>For more information, visit <a href="https://www.linkedin.com/in/paulinus-jua-21255116b/" target="_blank">Paulinus Jua LinkedIn</a>.</p>
                   <p>© 2025 Paulinus Jua. All rights reserved.</p>""",
    theme="peach",  
)

# Launch the app
if __name__ == "__main__":
    iface.launch()