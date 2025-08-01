import easyocr
import re

reader = easyocr.Reader(['en'])

def extract_text_from_image(image):
    results = reader.readtext(image)
    return " ".join([res[1] for res in results])

def parse_cbc_features(text):
    features = {
        "Age": 0, "Sex": 1, "Hb": 0, "Hct": 0, "MCV": 0,
        "MCH": 0, "MCHC": 0, "RDW": 0, "RBC count": 0
    }

    for key in features:
        pattern = rf"{key}[:\s]+([0-9\.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            features[key] = float(match.group(1))

    # Return in correct order for model
    return [
        features["Age"], features["Sex"], features["Hb"], features["Hct"],
        features["MCV"], features["MCH"], features["MCHC"],
        features["RDW"], features["RBC count"]
    ]
 
