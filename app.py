import gradio as gr
import os
import io
import json
import zipfile
import uuid
from datetime import datetime
from google.cloud import storage, vision, bigquery
from fastai.vision.all import Learner, ImageDataLoaders,load_learner, zipfile, Resize, aug_transforms, vision_learner, load_model, resnet34, Image, accuracy,PILImage
from pathlib import Path

#Setting up GCP client
credentials_content = os.environ['gcp_cam']
with open('gcp_key.json', 'w') as f:
    f.write(credentials_content)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_key.json'
bucket_name = os.environ['gcp_bucket']
pkl_blob = os.environ['pretrained_model'] 
dataset_blob = os.environ['dataset_path'] 
local_pkl = Path('cam_food_model.pkl')
local_zip = 'dataset.zip'
local_dataset_path = Path('dataset')
bq_client = bigquery.Client()
bq_dataset = os.environ['bq_dataset'] 
bq_table = os.environ['bq_table'] 
#image_upload_bucket = 'my-app-user-images'
upload_folder = os.environ['user_data_gcp'] 


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



#learn = vision_learner(dls, resnet34, metrics=accuracy)
#load_model(local_pkl, learn.model, learn.opt)
learn = load_learner(local_pkl)


def upload_image_to_gcs(local_path, dest_folder, dest_filename):
    blob = bucket.blob(f"{upload_folder}/{dest_folder}{dest_filename}")
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{upload_folder}/{dest_folder}{dest_filename}"

def log_to_bigquery(record):
    table_id = f"{bq_client.project}.{bq_dataset}.{bq_table}"
    errors = bq_client.insert_rows_json(table_id, [record])
    if errors:
        print("BigQuery insert errors:", errors)




def resize_image(img_path, max_width=640, max_height=480):
    img = Image.open(img_path)
    img.thumbnail((max_width, max_height))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    return buf



generic_terms_str = os.getenv('generic_terms', '')
generic_terms = set(generic_terms_str.split(',')) 

def call_google_food_api(image_path):
    try:
        client = vision.ImageAnnotatorClient()
        with open(image_path, 'rb') as img_file:
            content = img_file.read()
        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        
        if response.error.message:
            return f"Error with Vision API: {response.error.message}"

        labels = response.label_annotations
        food_labels = [
            label.description for label in labels
            if 'food' in label.description.lower() or 'dish' in label.description.lower()
        ]

        # Filter out generic terms
        specific_labels = [label for label in food_labels if label.lower() not in generic_terms]

        if specific_labels:
            return f"Not Rendered in Our Model \n\nGoogle detected: {specific_labels[0]}"
        else:
            return "Google detected: Unknown food"

    except Exception as e:
        return f"Error calling Google Vision API: {str(e)}"









def predict(image_path, threshold=0.40):
    unique_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    img = PILImage.create(image_path)
    resized_img = resize_image(img)

    pred_class, pred_idx, outputs = learn.predict(resized_img)
    prob = outputs[pred_idx].item()

    # Decide folder
    dest_folder = f"user_data/{pred_class}/" if prob >= threshold else "user_data/unknown/"

    uploaded_gcs_path = upload_image_to_gcs(image_path, dest_folder, f"{unique_id}.jpg")

    log_to_bigquery({
        "id": unique_id,
        "timestamp": timestamp,
        "image_gcs_path": uploaded_gcs_path,
        "predicted_class": pred_class,
        "confidence": prob,
        "threshold": threshold
    })

    if prob >= threshold:
        return f"Meal: {pred_class}, Confidence: {prob:.4f}"
    else:
        return f"Unknown Meal, Confidence: {prob:.4f}"




#Build Gradio interface


def unified_predict(upload_files, webcam_img, clipboard_img):
    results = []
    if upload_files:
        results = [predict(file.name) for file in upload_files]
    elif webcam_img:
        results = [predict(webcam_img)]
    elif clipboard_img:
        results = [predict(clipboard_img)]
    else:
        return "No image provided."
    return "\n\n".join(results)

with gr.Blocks(theme="peach") as demo:
    gr.Markdown("""# Cameroonian Meal Recognizer  
    <p><b>Welcome to Version 1:</b> Identify traditional Cameroonian dishes from a photo.</p>
    <p><mark>This tool offers a friendly playground to learn about our diverse dishes. Therefore multiple image upload is encouraged for improvement in subsequent versions predictions.</mark></p>
    <p><i>Choose an input source below, and our AI will recognize the meal.</i></p>
    """)

    with gr.Tabs():
        with gr.Tab("Upload"):
            upload_input = gr.File(file_types=["image"], file_count="multiple", label="Upload Meal Images")
        with gr.Tab("Webcam"):
            webcam_input = gr.Image(type="filepath", sources=["webcam"], label="Capture from Webcam")
        with gr.Tab("Clipboard"):
            clipboard_input = gr.Image(type="filepath", sources=["clipboard"], label="Paste from Clipboard")

    submit_btn = gr.Button("Identify Meal")
    output_box = gr.Textbox(label="Prediction Result", lines=10)

    submit_btn.click(
        fn=unified_predict,
        inputs=[upload_input, webcam_input, clipboard_input],
        outputs=output_box
    )

    gr.Markdown("""
    <p>Future updates will include:
    <ul>
        <li>Ingredient lists</li>
        <li>Meal preparation details</li>
        <li>Origin (locality) info</li>
        <li>Nearby restaurants</li>
    </ul></p>
    <p>Learn more on <a href="https://www.linkedin.com/in/paulinus-jua-21255116b/" target="_blank">Paulinus Jua's LinkedIn</a>.</p>
    <p>© 2025 Paulinus Jua. All rights reserved.</p>
    """)

if __name__ == "__main__":
    demo.launch()



