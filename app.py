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
from transformers import pipeline

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
def predict(image_path, threshold=0.30, user_feedback=None):
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

    print(f"Prediction time: {time.time() - start_time:.2f}s")

    return (
    f"❓ Unknown Meal: Provide Name. Thanks" if prob <= threshold else
    f"⚠️ Meal: {pred_class}, Low Confidence" if 0.275 <= prob <= 0.5 else
    f"✅ Meal: {pred_class}"
)




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
# Load translator once
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# Simple translation function
def translate(text, lang):
    if lang == "fr":
        return translator(text, max_length=512)[0]["translation_text"]
    return text  # default English

# UI builder function with dynamic language
def build_interface(selected_lang):
    title = translate("Cameroonian Meal Recognizer", selected_lang)
    welcome = translate("Welcome to Version 1:", selected_lang)
    instruction = translate("Identify traditional Cameroonian dishes from a photo.", selected_lang)
    guide = translate("Choose an input source below, and our AI will recognize the meal.", selected_lang)
    upload_label = translate("Upload Meal Images", selected_lang)
    webcam_label = translate("Capture from Webcam", selected_lang)
    clipboard_label = translate("Paste from Clipboard", selected_lang)
    feedback_label = translate("Feedback: If the prediction is wrong, enter the correct meal name", selected_lang)
    predict_btn = translate("Identify Meal", selected_lang)
    output_label = translate("Prediction Result", selected_lang)

    with gr.Blocks(theme="peach", analytics_enabled=False) as demo:
        gr.Markdown(f"""# {title}
        <p><b>{welcome}</b> {instruction}</p>
        <p style='background-color: #b3e5fc; padding: 5px; border-radius: 4px;'>
        This tool offers a friendly playground to learn about our diverse dishes.
        Therefore multiple image upload is encouraged for improvement in subsequent versions predictions.</p>
        <p><i>{guide}</i></p>""")

        with gr.Tabs():
            with gr.Tab(translate("Upload", selected_lang)):
                upload_input = gr.File(file_types=["image"], file_count="multiple", label=upload_label)
            with gr.Tab(translate("Webcam", selected_lang)):
                webcam_input = gr.Image(type="filepath", sources=["webcam"], label=webcam_label)
            with gr.Tab(translate("Clipboard", selected_lang)):
                clipboard_input = gr.Image(type="filepath", sources=["clipboard"], label=clipboard_label)

        feedback_input = gr.Textbox(label=feedback_label)
        submit_btn = gr.Button(predict_btn)
        output_box = gr.Textbox(label=output_label, lines=10)

        submit_btn.click(
            fn=unified_predict,
            inputs=[upload_input, webcam_input, clipboard_input, feedback_input],
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

    return demo

# Dropdown wrapper
def launch_app(lang_choice):
    return build_interface(lang_choice)

# Top-level selector
with gr.Blocks() as wrapper:
    lang_dropdown = gr.Dropdown(
        choices=[("English", "en"), ("Français", "fr")],
        label="🌐 Select Language",
        value="en"
    )
    lang_btn = gr.Button("Load App")

    app_output = gr.update(visible=False)

    def launch_selected(lang):
        demo = build_interface(lang)
        demo.launch(share=False)  # launch inside callback
        return gr.update(visible=True)

    lang_btn.click(fn=launch_selected, inputs=lang_dropdown, outputs=app_output)

if __name__ == "__main__":
    wrapper.launch()





