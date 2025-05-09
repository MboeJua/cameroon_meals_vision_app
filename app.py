import gradio as gr
import os
import json
from google.cloud import storage
from fastai.vision.all import Learner, PILImage
from pathlib import Path


#Setting up GCP client
credentials_content = os.environ['gcp_cam']
with open('gcp_key.json', 'w') as f:
    f.write(credentials_content)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_key.json'
bucket_name = os.environ['gcp_bucket']
pkl_blob = 'paulinus/cameroon_food.pkl'
local_pkl = Path('cameroon_food.pkl')

client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(pkl_blob)
blob.download_to_filename(local_pkl)

#Load model
learn = Learner.load(local_pkl)



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