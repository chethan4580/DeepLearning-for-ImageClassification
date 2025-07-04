# 📌 Step 1: Install required packages
!pip install tensorflow ipywidgets --quiet

# 📌 Step 2: Import modules
import os
import io
from PIL import Image as PILImage
from IPython.display import display, HTML
import ipywidgets as widgets
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# 📌 Step 3: Load pretrained model
model = MobileNetV2(weights='imagenet')

# 📁 Step 4: Output folder for saving results
output_dir = "classified_images"
os.makedirs(output_dir, exist_ok=True)

# 📌 Step 5: Upload widget for multiple files
uploader = widgets.FileUpload(accept='image/*', multiple=True)

# 📌 Step 6: Classification function with two-row output
def classify_images(change):
    for filename in uploader.value:
        content = uploader.value[filename]['content']
        image = PILImage.open(io.BytesIO(content)).convert('RGB')

        # Preprocess
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))

        # Predict
        preds = model.predict(img_array)
        label, confidence = decode_predictions(preds, top=1)[0][0][1:3]

        # 📸 Display image
        display(image.resize((150, 150)))

        # 🧠 Display prediction in two rows
        display(HTML(f"""
            <div style="font-family:Arial; line-height:1.5; font-size:16px;">
                <strong>Prediction:</strong><br>
                {label}<br>
                {confidence * 100:.2f}%
            </div>
        """))

        # 💾 Save with label & confidence
        safe_label = label.replace(" ", "_")
        save_name = f"{safe_label}_{confidence*100:.2f}_{filename}"
        image.save(os.path.join(output_dir, save_name))

# 📌 Step 7: Bind function and show widget
uploader.observe(classify_images, names='value')
display(HTML("<h3>📂 Upload Multiple Images</h3><p>Predictions will appear below each image.</p>"))
display(uploader)
