import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import gradio as gr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.model import get_model
from src.dataset import val_transform


# ─── Load Model ──────────────────────────────────────────
model = get_model()
model_path = os.path.join(config.MODELS_PATH, 'best_model.pth')

if not os.path.exists(model_path):
    raise FileNotFoundError('❌ No trained model found! Run train.py first.')

model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
model.eval()
print('✅ Model loaded for demo!')


# ─── Inference Function ──────────────────────────────────
def predict_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total == 0:
            return 'Error: Could not read video file', None

        indices = np.linspace(0, total - 1, config.NUM_FRAMES, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = val_transform(img)
                frames.append(img)
        cap.release()

        if len(frames) == 0:
            return 'Error: No frames extracted', None

        # Pad if needed
        if len(frames) < config.NUM_FRAMES:
            frames += [frames[-1]] * (config.NUM_FRAMES - len(frames))

        # Shape: (1, num_frames, C, H, W)
        input_tensor = torch.stack(frames).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probs  = torch.softmax(output, dim=1)[0]
            fake_prob = probs[1].item()
            real_prob = probs[0].item()

        if fake_prob > 0.5:
            verdict = f'🚨 FAKE — {fake_prob*100:.1f}% confidence'
        else:
            verdict = f'✅ REAL — {real_prob*100:.1f}% confidence'

        return verdict, {'REAL': round(real_prob, 3), 'FAKE': round(fake_prob, 3)}

    except Exception as e:
        return f'Error: {str(e)}', None


# ─── Gradio Interface ────────────────────────────────────
demo = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label='Upload a Video'),
    outputs=[
        gr.Textbox(label='Verdict'),
        gr.Label(label='Confidence Scores')
    ],
    title='🎭 Deepfake Detector',
    description='Upload a video to detect whether it is REAL or FAKE using CNN + LSTM spatial-temporal analysis.',
)

if __name__ == '__main__':
    demo.launch(share=True)