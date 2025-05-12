import gradio as gr
import os
import io
import json
import zipfile
import uuid
from datetime import datetime
#from PIL import Image as PILImages
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
    img.save(buf, format='JPEG', quality=70)
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





def predict(files, threshold=0.40):
    # If only one file is uploaded, wrap it in a list
    if not isinstance(files, list):
        files = [files]
        
    results = []
    for file in files:
        img = PILImage.create(file)
        unique_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        resized_img = resize_image(img)
        pred_class, pred_idx, outputs = learn.predict((resized_img))
        prob = outputs[pred_idx].item()

        dest_folder = f"user_data/{pred_class}/" if prob >= threshold else "user_data/unknown/"
        uploaded_gcs_path = upload_image_to_gcs(img, dest_folder, f"{unique_id}.jpg")

        log_to_bigquery({
            "id": unique_id,
            "timestamp": timestamp,
            "image_gcs_path": uploaded_gcs_path,
            "predicted_class": pred_class,
            "confidence": prob,
            "threshold": threshold
        })

        results.append({
            "Image": os.path.basename(file.name) if hasattr(file, 'name') else "Captured Image",
            "Prediction": pred_class if prob >= threshold else "Unknown",
            "Confidence": round(prob, 4)
        })
    return results
    
        # Low confidence → call Google API
        #google_result = call_google_food_api(img)




#Build Gradio interface
with gr.Blocks(title="Cameroonian Meal Recognizer") as demo:
    gr.HTML("""
        <h2>Discover Authentic Cameroonian Meals!</h2>
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
        <p><b>You are kindly requested to submit multiple images, as this will help improve the accuracy of future versions through retraining.</b></p>
        <p>© 2025 Paulinus Jua. All rights reserved.</p>
    """)

    with gr.Tab("Upload Multiple Images"):
        file_input = gr.File(file_types=["image"], label="Upload images")
        output_multi = gr.Dataframe(headers=["Image", "Prediction", "Confidence"])
        file_input.change(fn=predict, inputs=file_input, outputs=output_multi)

    with gr.Tab("Webcam or Clipboard (Single Image)"):
        single_input = gr.Image(type="pil", sources=["webcam", "clipboard"], label="Capture or paste an image")
        output_single = gr.Dataframe(headers=["Image", "Prediction", "Confidence"])
        single_input.change(fn=lambda img: predict([img]), inputs=single_input, outputs=output_single)

if __name__ == "__main__":
    demo.launch()
