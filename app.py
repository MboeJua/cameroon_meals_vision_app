import gradio as gr
import os
import json
from fastai.vision.all import load_learner, PILImage

gcp_json = json.loads(os.environ["gcp_cam"])

learn = load_learner("cameroon_food.pkl")


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