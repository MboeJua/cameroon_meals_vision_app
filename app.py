import gradio as gr
import os
import uuid
import time
from datetime import datetime
from threading import Thread
from google.cloud import storage, bigquery
from fastai.vision.all import load_learner, PILImage
from pathlib import Path

credentials_content = os.environ['gcp_cam']
with open('gcp_key.json', 'w') as f:
    f.write(credentials_content)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_key.json'

bucket_name = os.environ['gcp_bucket']
pkl_blob = os.environ['pretrained_model']
upload_folder = os.environ['user_data_gcp']
bq_dataset = os.environ['bq_dataset']
bq_table = os.environ['bq_table']

#Load Model
local_pkl = Path('cam_food_model.pkl')
if not local_pkl.exists():
    storage.Client().bucket(bucket_name).blob(pkl_blob).download_to_filename(local_pkl)
learn = load_learner(local_pkl)


bq_client = bigquery.Client()
bucket = storage.Client().bucket(bucket_name)

#Async Logging
def log_to_bigquery(record):
    table_id = f"{bq_client.project}.{bq_dataset}.{bq_table}"
    try:
        errors = bq_client.insert_rows_json(table_id, [record])
        if errors:
            print("BigQuery insert errors:", errors)
    except Exception as e:
        print("Logging error:", e)

def async_log(record):
    Thread(target=log_to_bigquery, args=(record,), daemon=True).start()

def log_feedback_to_bigquery(record):
    table_id = f"{bq_client.project}.{bq_dataset}.{bq_table}_feedback"
    try:
        errors = bq_client.insert_rows_json(table_id, [record])
        if errors:
            print("Feedback insert errors:", errors)
    except Exception as e:
        print("Feedback logging error:", e)

def async_feedback_log(record):
    Thread(target=log_feedback_to_bigquery, args=(record,), daemon=True).start()

#GCS Upload
def upload_image_to_gcs(local_path, dest_folder, dest_filename):
    blob = bucket.blob(f"{upload_folder}/{dest_folder}{dest_filename}")
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{upload_folder}/{dest_folder}{dest_filename}"

# Prediction
def predict(image_path, threshold=0.40):
    unique_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    img = PILImage.create(image_path)
    pred_class, pred_idx, outputs = learn.predict(img)
    prob = outputs[pred_idx].item()

    dest_folder = f"user_data/{pred_class}/" if prob >= threshold else "user_data/unknown/"
    uploaded_gcs_path = upload_image_to_gcs(image_path, dest_folder, f"{unique_id}.jpg")

    async_log({
        "id": unique_id,
        "timestamp": timestamp,
        "image_gcs_path": uploaded_gcs_path,
        "predicted_class": pred_class,
        "confidence": prob,
        "threshold": threshold
    })

    return f"Meal: {pred_class}, Confidence: {prob:.4f}" if prob >= threshold else f"Unknown Meal, Confidence: {prob:.4f}"

# Gradio Unified Predict
def unified_predict(upload_files, webcam_img, clipboard_img):
    files = []
    if upload_files:
        files = [file.name for file in upload_files]
    elif webcam_img:
        files = [webcam_img]
    elif clipboard_img:
        files = [clipboard_img]
    else:
        return "No image provided."

    return "\n\n".join([predict(f) for f in files])

# Feedback Function
def submit_feedback(pred_result, user_input):
    if not user_input.strip():
        return "No feedback entered."

    feedback_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    feedback_record = {
        "id": feedback_id,
        "timestamp": timestamp,
        "prediction_result": pred_result,
        "user_feedback": user_input
    }

    async_feedback_log(feedback_record)
    return "✅ Thank you! Feedback received."

# Gradio UI
with gr.Blocks(theme="peach", analytics_enabled=False) as demo:
    gr.Markdown("""# Cameroonian Meal Recognizer  
    <p><b>Welcome to Version 1:</b> Identify traditional Cameroonian dishes from a photo.</p>
    <p style='background-color: #b3e5fc; padding: 5px; border-radius: 4px;'>This tool offers a friendly playground to learn about our diverse dishes. Therefore multiple image upload is encouraged for improvement in subsequent versions predictions.</p>
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

    feedback_text = gr.Textbox(label="If incorrect, type the actual meal name (optional):")
    feedback_btn = gr.Button("Submit Feedback")
    feedback_status = gr.Textbox(label="Feedback Status", interactive=False)

    submit_btn.click(
        fn=unified_predict,
        inputs=[upload_input, webcam_input, clipboard_input],
        outputs=output_box
    )

    feedback_btn.click(
        fn=submit_feedback,
        inputs=[output_box, feedback_text],
        outputs=feedback_status
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, ssr_mode=False)





