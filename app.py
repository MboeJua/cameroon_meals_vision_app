import gradio as gr
import os
import uuid
import time
from datetime import datetime
from threading import Thread
from google.cloud import storage, bigquery
from fastai.vision.all import load_learner, PILImage
from fastai.vision.augment import Resize  
from pathlib import Path
from collections import deque

# Setup GCP credentials
credentials_content = os.environ['gcp_cam']
with open('gcp_key.json', 'w') as f:
    f.write(credentials_content)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_key.json'

# GCP config
bucket_name = os.environ['gcp_bucket']
pkl_blob = os.environ['pretrained_model']
upload_folder = os.environ['user_data_gcp']
bq_dataset = os.environ['bq_dataset']
bq_table = os.environ['bq_table']

# Load model
local_pkl = Path('cam_meals_f2.pkl')
if not local_pkl.exists():
    storage.Client().bucket(bucket_name).blob(pkl_blob).download_to_filename(local_pkl)

learn = load_learner(local_pkl)
bq_client = bigquery.Client()
bucket = storage.Client().bucket(bucket_name)

# Store recent prediction IDs
deferred_feedback = deque(maxlen=100)  

# Upload image to GCS
def upload_image_to_gcs(local_path, dest_folder, dest_filename):
    blob = bucket.blob(f"{upload_folder}/{dest_folder}{dest_filename}")
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{upload_folder}/{dest_folder}{dest_filename}"

# Async BigQuery logging
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

# Prediction logic with feedback
def predict(image_path, threshold=0.275, user_feedback=None):
    start_time = time.time()
    unique_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Load and resize image using fastai's PILImage
    try:
        img = PILImage.create(image_path)
        img = img.resize((256, 256))  
    except Exception as e:
        print("Image processing error:", e)
        return "Image could not be processed."

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
        "threshold": threshold,
        "user_feedback": user_feedback or ""
    })
    deferred_feedback.append((time.time(), unique_id))

    print(f"Prediction time: {time.time() - start_time:.2f}s")

    return (
    f"❓ Unknown Meal: Provide Name. Thanks" if prob <= threshold else
    f"⚠️ Meal: {pred_class}, Low Confidence" if 0.275 <= prob <= 0.5 else
    f"✅ Meal: {pred_class}"
)

# Feedback-only logic
def submit_feedback_only(feedback_text):
    if not feedback_text.strip():
        return "⚠️ No feedback provided."

    now = time.time()
    for ts, uid in reversed(deferred_feedback):
        if now - ts <= 120:
            async_log({
                "id": uid,
                "timestamp": datetime.utcnow().isoformat(),
                "image_gcs_path": "feedback_only",
                "predicted_class": "feedback_update",
                "confidence": None,
                "threshold": None,
                "user_feedback": feedback_text
            })
            return "✅ Feedback linked to recent prediction. Thank you!"

    return "⚠️ Feedback not linked: time expired."


# Handle multiple images + feedback
def unified_predict(upload_files, webcam_img, clipboard_img, feedback):
    files = []
    if upload_files:
        files = [file.name for file in upload_files]
    elif webcam_img:
        files = [webcam_img]
    elif clipboard_img:
        files = [clipboard_img]
    else:
        return "No image provided."

    return "\n\n".join([predict(f, user_feedback=feedback) for f in files])

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
    output_box = gr.Textbox(label="Prediction Result", lines=6)
    
    gr.Markdown("### Feedback")

    with gr.Row():
        feedback_input = gr.Textbox(
            label=None,
            placeholder="If prediction is wrong, enter correct meal name...",
            lines=1,
            scale=4
        )
        feedback_btn = gr.Button("Submit Feedback", scale=1)

    feedback_ack = gr.Textbox(visible=False, interactive=False)

    submit_btn.click(
        fn=unified_predict,
        inputs=[upload_input, webcam_input, clipboard_input, feedback_input],
        outputs=output_box
    )

    def styled_feedback_msg(feedback_text):
        msg = submit_feedback_only(feedback_text)
        if msg.startswith("✅"):
            return f"<span style='color: green; font-weight: bold;'>{msg}</span>"
        elif msg.startswith("⚠️"):
            return f"<span style='color: orange; font-weight: bold;'>{msg}</span>"
        return msg

    feedback_btn.click(
        fn=submit_feedback_only,
        inputs=feedback_input,
        outputs=feedback_ack
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
    print("App setup complete — launching Gradio...")
    demo.launch()
    print("Launched.")





