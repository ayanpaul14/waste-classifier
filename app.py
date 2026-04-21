import os
import json
import time
import math
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
WASTE_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ── Optional: auto-download weights from a URL at startup ────────────────
# Upload your .pth file to Google Drive, Dropbox, or GitHub Releases,
# then set this env var on Render:  MODEL_URL=https://...
# Leave blank to skip download.
MODEL_DOWNLOAD_URL = os.environ.get('MODEL_URL', '')
WEIGHTS_PATH = 'models/waste_classifier.pth'

CLASS_INFO = {
    'cardboard': {
        'icon': '📦', 'bin': 'Blue recycling bin',
        'tip': 'Flatten before disposal. Remove tape and staples.',
        'color': '#8B6914', 'recyclable': True,
        'process': [
            {'step': 1, 'title': 'Remove contaminants', 'desc': 'Take out any plastic liners, bubble wrap, or foam inserts.', 'icon': '🧹'},
            {'step': 2, 'title': 'Remove tape and staples', 'desc': 'Peel off all adhesive tape and remove staples — these contaminate the batch.', 'icon': '📌'},
            {'step': 3, 'title': 'Flatten the box', 'desc': 'Break down and flatten the cardboard to save space in the recycling bin.', 'icon': '📐'},
            {'step': 4, 'title': 'Keep it dry', 'desc': 'Wet cardboard loses its fibres and cannot be recycled. Store in a dry place.', 'icon': '💧'},
            {'step': 5, 'title': 'Place in recycling bin', 'desc': 'It will be pulped, cleaned, and turned into new paper products.', 'icon': '♻️'}
        ],
        'fun_fact': 'Recycling one tonne of cardboard saves 17 trees and 26,500 litres of water.',
        'recycling_rate': 85
    },
    'glass': {
        'icon': '🫙', 'bin': 'Green glass bank',
        'tip': 'Rinse thoroughly. Do not mix with ceramics or mirrors.',
        'color': '#2E7D32', 'recyclable': True,
        'process': [
            {'step': 1, 'title': 'Empty and rinse', 'desc': 'Rinse bottles and jars with water to remove food or drink residue.', 'icon': '🚿'},
            {'step': 2, 'title': 'Remove lids and caps', 'desc': 'Metal lids go with metals. Plastic caps go in the plastic bin.', 'icon': '🔩'},
            {'step': 3, 'title': 'Do not break', 'desc': 'Keep glass intact. Broken glass is hazardous to recycling workers.', 'icon': '⚠️'},
            {'step': 4, 'title': 'Check what is accepted', 'desc': 'Only bottles and jars. Mirrors, window glass, and ceramics are NOT accepted.', 'icon': '✅'},
            {'step': 5, 'title': 'Drop at glass bank', 'desc': 'Glass is 100% recyclable and can be recycled endlessly without quality loss.', 'icon': '🏦'}
        ],
        'fun_fact': 'Glass can be recycled endlessly without any loss in quality or purity.',
        'recycling_rate': 76
    },
    'metal': {
        'icon': '🥫', 'bin': 'Blue recycling bin',
        'tip': 'Rinse cans. Aluminium and steel are both recyclable.',
        'color': '#546E7A', 'recyclable': True,
        'process': [
            {'step': 1, 'title': 'Empty and rinse', 'desc': 'Empty the can completely and give it a quick rinse to avoid contamination.', 'icon': '🚿'},
            {'step': 2, 'title': 'Identify the metal', 'desc': 'Use a magnet — steel sticks, aluminium does not. Both are recyclable.', 'icon': '🧲'},
            {'step': 3, 'title': 'Do not crush aluminium cans', 'desc': 'Some facilities need cans uncrushed for automated sorting.', 'icon': '🔄'},
            {'step': 4, 'title': 'Remove non-metal parts', 'desc': 'Remove plastic lids if they come off easily.', 'icon': '🏷️'},
            {'step': 5, 'title': 'Place in recycling bin', 'desc': 'Metals are melted and reformed — saving up to 95% of energy vs new metal.', 'icon': '♻️'}
        ],
        'fun_fact': 'Recycling aluminium uses 95% less energy than producing it from raw ore.',
        'recycling_rate': 67
    },
    'paper': {
        'icon': '📄', 'bin': 'Blue recycling bin',
        'tip': 'Keep dry. Shredded paper goes in a sealed bag.',
        'color': '#F57F17', 'recyclable': True,
        'process': [
            {'step': 1, 'title': 'Check if it is clean', 'desc': 'Paper with heavy food grease cannot be recycled. Tear off the soiled part, recycle the rest.', 'icon': '🔍'},
            {'step': 2, 'title': 'Remove plastic elements', 'desc': 'Remove plastic windows from envelopes and any plastic coatings.', 'icon': '🪟'},
            {'step': 3, 'title': 'Keep it dry', 'desc': 'Wet paper fibres break down and cannot be recycled. Store in a dry location.', 'icon': '☀️'},
            {'step': 4, 'title': 'Shredded paper', 'desc': 'Put shredded paper in a sealed paper bag or tied bundle to avoid jamming machinery.', 'icon': '📋'},
            {'step': 5, 'title': 'Place in paper recycling', 'desc': 'Paper is pulped with water, cleaned, and rolled into new sheets. Recyclable 5–7 times.', 'icon': '♻️'}
        ],
        'fun_fact': 'One tonne of recycled paper saves 17 trees and uses 70% less energy than making new paper.',
        'recycling_rate': 68
    },
    'plastic': {
        'icon': '🧴', 'bin': 'Blue recycling bin',
        'tip': 'Check the resin code (1-7). Rinse containers before recycling.',
        'color': '#1565C0', 'recyclable': True,
        'process': [
            {'step': 1, 'title': 'Find the resin code', 'desc': 'Look for the triangle symbol with a number (1-7). Codes 1 (PET) and 2 (HDPE) are most widely accepted.', 'icon': '🔢'},
            {'step': 2, 'title': 'Empty and rinse', 'desc': 'Rinse containers to remove food or liquid residue before recycling.', 'icon': '🚿'},
            {'step': 3, 'title': 'Remove non-plastic parts', 'desc': 'Remove metal lids and put them in the metal bin.', 'icon': '🔩'},
            {'step': 4, 'title': 'Do not bag recyclables', 'desc': 'Never put recyclables inside a plastic bag — the bag jams sorting machines.', 'icon': '🚫'},
            {'step': 5, 'title': 'Place in recycling bin', 'desc': 'Plastics are sorted, shredded, melted, and moulded into new products like fleece clothing.', 'icon': '♻️'}
        ],
        'fun_fact': 'A recycled plastic bottle can become a fleece jacket in as little as 30 days.',
        'recycling_rate': 44
    },
    'trash': {
        'icon': '🗑️', 'bin': 'Black general waste bin',
        'tip': 'Non-recyclable waste. Consider if any part can be separated.',
        'color': '#424242', 'recyclable': False,
        'process': [
            {'step': 1, 'title': 'Double-check recyclability', 'desc': 'Before binning, check if any part is recyclable — cardboard wrapping, metal, or glass.', 'icon': '🔍'},
            {'step': 2, 'title': 'Check for special disposal', 'desc': 'Batteries, electronics, medicines, and chemicals need special disposal — not general waste.', 'icon': '⚠️'},
            {'step': 3, 'title': 'Consider reuse', 'desc': 'Can the item be donated, repaired, or repurposed? Charity shops and Freecycle are great options.', 'icon': '🔄'},
            {'step': 4, 'title': 'Reduce future waste', 'desc': 'Choose products with less packaging and opt for reusable alternatives where possible.', 'icon': '🌱'},
            {'step': 5, 'title': 'Dispose in general waste', 'desc': 'Place in the black bin. General waste goes to landfill or energy-from-waste facilities.', 'icon': '🗑️'}
        ],
        'fun_fact': 'The average person generates about 1.5 kg of waste per day. Small habit changes make a big difference.',
        'recycling_rate': 0
    }
}


# ── Helpers ──────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_float(value):
    """Convert tensor value to float, replacing nan/inf with 0."""
    v = float(value)
    return 0.0 if (math.isnan(v) or math.isinf(v)) else v


def maybe_download_weights():
    """Download model weights from MODEL_URL env var if file is missing."""
    if os.path.exists(WEIGHTS_PATH):
        return
    if not MODEL_DOWNLOAD_URL:
        print("No MODEL_URL set and no local weights found — running in demo mode.")
        return
    print(f"Downloading model weights from {MODEL_DOWNLOAD_URL} ...")
    os.makedirs('models', exist_ok=True)
    try:
        r = requests.get(MODEL_DOWNLOAD_URL, timeout=120)
        r.raise_for_status()
        with open(WEIGHTS_PATH, 'wb') as f:
            f.write(r.content)
        print(f"Weights saved to {WEIGHTS_PATH}")
    except Exception as e:
        print(f"Failed to download weights: {e}  — running in demo mode.")


def load_model():
    maybe_download_weights()
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(WASTE_CLASSES))
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        print("✅ Loaded fine-tuned weights.")
    else:
        print("⚠️  No weights found — predictions will be random (demo mode).")
    model.eval()
    return model


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = load_model()


def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        start = time.time()
        outputs = model(tensor)
        inference_time = (time.time() - start) * 1000

    # ── FIX: guard against nan/inf before softmax ──────────────────────
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        outputs = torch.zeros_like(outputs)

    probabilities = torch.softmax(outputs, dim=1)[0]
    confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = WASTE_CLASSES[predicted_idx.item()]
    info = CLASS_INFO[predicted_class]

    # ── FIX: safe_float prevents json.dumps from crashing on nan ───────
    conf_val = safe_float(confidence) * 100

    all_probs = {
        cls: round(safe_float(probabilities[i]) * 100, 1)
        for i, cls in enumerate(WASTE_CLASSES)
    }

    # Normalise probabilities to sum to 100 (they should, but guard anyway)
    total = sum(all_probs.values())
    if total > 0 and abs(total - 100) > 0.5:
        all_probs = {k: round(v / total * 100, 1) for k, v in all_probs.items()}

    return {
        'predicted_class': predicted_class,
        'confidence': round(conf_val, 1),
        'bin': info['bin'],
        'tip': info['tip'],
        'icon': info['icon'],
        'color': info['color'],
        'recyclable': info['recyclable'],
        'process': info['process'],
        'fun_fact': info['fun_fact'],
        'recycling_rate': info['recycling_rate'],
        'inference_time_ms': round(inference_time, 1),
        'all_probabilities': all_probs,
        'demo_mode': not os.path.exists(WEIGHTS_PATH)
    }


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Use PNG, JPG, or WEBP.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    try:
        result = predict_image(filepath)
        result['filename'] = filename
        # ── FIX: explicitly set content type and use json.dumps ────────
        response_body = json.dumps(result, ensure_ascii=False)
        return app.response_class(
            response=response_body,
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/classes')
def get_classes():
    return jsonify({'classes': WASTE_CLASSES, 'info': CLASS_INFO})


@app.route('/health')
def health():
    weights_loaded = os.path.exists(WEIGHTS_PATH)
    return jsonify({
        'status': 'ok',
        'weights_loaded': weights_loaded,
        'demo_mode': not weights_loaded
    }), 200


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)