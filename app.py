from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import mediapipe as mp
import pickle
import os
import random

app = Flask(__name__)

# =========================
# LOAD MODELS
# =========================
arabic_model = tf.saved_model.load("arabic_saved_model")
english_model = tf.saved_model.load("english_saved_model")

# =========================
# LOAD ENGLISH LABEL ENCODER
# =========================
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

english_classes = list(le.classes_)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# =========================
# ARABIC CLASSES
# =========================
arabic_classes = [
    "ain", "al", "aleff", "bb", "dal", "dha", "dhad",
    "fa", "gaaf", "ghain", "ha", "haa", "jeem", "kaaf",
    "khaa", "la", "laam", "meem", "nun", "ra", "saad",
    "seen", "sheen", "ta", "taa", "thaa", "thal",
    "toot", "waw", "ya", "yaa", "zay"
]

IMG_SIZE = (224, 224)

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# IMAGE TO TEXT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    language = request.form.get("language", "ar")

    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # =========================
    # ENGLISH MODEL
    # =========================
    if language == "en":

        img_np = np.array(image)

        result = hands.process(img_np)

        if not result.multi_hand_landmarks:
            return jsonify({
                "prediction": "No hand detected",
                "confidence": 0
            })

        hand = result.multi_hand_landmarks[0]

        landmarks = []

        # معرفة نوع اليد
        hand_label = result.multi_handedness[0].classification[0].label

        for lm in hand.landmark:

            # نعكس فقط إذا اليد اليسار
            if hand_label == "Left":
                x = 1 - lm.x
            else:
                x = lm.x

            y = lm.y
            z = lm.z

            landmarks.extend([x, y, z])

        landmarks = np.array(
            landmarks,
            dtype=np.float32
        ).reshape(1, 63)

        infer_en = english_model.signatures["serving_default"]

        prediction = infer_en(
            tf.convert_to_tensor(
                landmarks,
                dtype=tf.float32
            )
        )

        prediction = list(
            prediction.values()
        )[0].numpy()

        predicted_index = int(np.argmax(prediction))

        predicted_class = english_classes[predicted_index]

        confidence = float(np.max(prediction))

    # =========================
    # ARABIC MODEL
    # =========================
    else:

        image = image.resize(IMG_SIZE)

        img_array = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        infer_ar = arabic_model.signatures["serving_default"]

        prediction = infer_ar(
            tf.convert_to_tensor(
                img_array,
                dtype=tf.float32
            )
        )

        prediction = list(
            prediction.values()
        )[0].numpy()

        predicted_index = int(np.argmax(prediction))

        predicted_class = arabic_classes[predicted_index]

        arabic_letters = {

            "ain": "ع",
            "al": "ال",
            "aleff": "ا",

            "bb": "ب",

            "dal": "د",
            "dha": "ظ",
            "dhad": "ض",

            "fa": "ف",

            "gaaf": "ق",
            "ghain": "غ",

            "ha": "ه",
            "haa": "ح",

            "jeem": "ج",

            "kaaf": "ك",
            "khaa": "خ",

            "la": "لا",
            "laam": "ل",

            "meem": "م",

            "nun": "ن",

            "ra": "ر",

            "saad": "ص",
            "seen": "س",
            "sheen": "ش",

            "ta": "ط",
            "taa": "ت",

            "thaa": "ث",
            "thal": "ذ",

            "toot": "ة",

            "waw": "و",

            "ya": "ي",
            "yaa": "ئ",

            "zay": "ز"
        }

        arabic_letter = arabic_letters.get(
            predicted_class,
            ""
        )

        predicted_class = f"{predicted_class} | {arabic_letter}"

        confidence = float(np.max(prediction))

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    })

# =========================
# TEXT TO SIGN (ARABIC)
# =========================

ARB_DATA_DIR = "static/arb_data/arabic_letters/arabic_letters"

arabic_letter_map = {

    "ا": "aleff",
    "أ": "aleff",
    "إ": "aleff",
    "آ": "aleff",

    "ب": "bb",
    "ط": "ta",
    "ث": "thaa",
    "ج": "jeem",
    "ح": "haa",
    "خ": "khaa",

    "د": "dal",
    "ذ": "thal",
    "ر": "ra",
    "ز": "zay",

    "س": "seen",
    "ش": "sheen",
    "ص": "saad",
    "ض": "dhad",

    "ة": "toot",
    "ظ": "dha",

    "ع": "ain",
    "غ": "ghain",

    "ف": "fa",
    "ق": "gaaf",
    "ك": "kaaf",
    "ل": "laam",

    "م": "meem",
    "ن": "nun",
    "ه": "ha",
    "و": "waw",

    "ي": "ya",
    "ئ": "yaa",

    "ت": "taa",

    "لا": "la",
    "ال": "al"
}

@app.route("/text_to_sign_ar", methods=["POST"])
def text_to_sign_ar():

    text = request.json.get("text", "")

    image_paths = []

    i = 0

    while i < len(text):

        if text[i].isspace():
            i += 1
            continue

        if i + 1 < len(text) and text[i:i+2] == "لا":
            folder_name = arabic_letter_map.get("لا")
            i += 2

        else:
            folder_name = arabic_letter_map.get(text[i])
            i += 1

        if not folder_name:
            continue

        class_path = os.path.join(
            ARB_DATA_DIR,
            folder_name
        )

        if os.path.exists(class_path):

            images = os.listdir(class_path)

            images = [
                img for img in images
                if img.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            ]

            if len(images) > 0:

                img_name = random.choice(images)

                img_path = f"/static/arb_data/arabic_letters/arabic_letters/{folder_name}/{img_name}"

                image_paths.append(img_path)

    return jsonify({
        "images": image_paths
    })

# =========================
# TEXT TO SIGN (ENGLISH)
# =========================

ENG_DATA_DIR = "static/eng_data/dataset_eng"

@app.route("/text_to_sign_en", methods=["POST"])
def text_to_sign_en():

    text = request.json.get("text", "")

    image_paths = []

    for char in text:

        if char == " ":
            continue

        letter = char.upper()

        class_path = os.path.join(
            ENG_DATA_DIR,
            letter
        )

        if os.path.exists(class_path):

            images = os.listdir(class_path)

            images = [
                img for img in images
                if img.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            ]

            if len(images) > 0:

                img_name = random.choice(images)

                img_path = f"/static/eng_data/dataset_eng/{letter}/{img_name}"

                image_paths.append(img_path)

    return jsonify({
        "images": image_paths
    })

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)