import gradio as gr
import os
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
learn.load('cameroon_food_weights',file=local_pkl)


def predict(img):
    pred_class, pred_idx, outputs = learn.predict(PILImage.create(img))
    prob = outputs[pred_idx].item()
    return f"Class: {pred_class}, Probability: {prob:.4f}"

#Build Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="file"),
    outputs=gr.Textbox(),
    title="Cameroonian Meal Identifier",
    description="Upload a meal image and get the predicted class."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()