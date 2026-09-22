from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

REFERENCE_RANGES = {
    "personAge": (1, 120),
    "blood_glucose_random": (22, 800),
    "blood_urea": (7, 200),
    "serum_creatinine": (0.1, 12.0),
    "sodium": (100, 145),
    "potassium": (2.5, 7.5),
    "white_blood_cell": (4500, 15000),
}

GENDER_SPECIFIC_RANGES = {
    "male": {
        "hemoglobin": (7.0, 17.5),
        "packed_cell_volume": (19, 55),
        "red_blood_cell": (2.35, 5.65),
    },
    "female": {
        "hemoglobin": (7.0, 15.5),
        "packed_cell_volume": (19, 50),
        "red_blood_cell": (2.35, 5.13),
    },
}

try:
    with open("models/CKD_final_model_SVM.sav", "rb") as f:
        model = pickle.load(f)

    with open("models/scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except (FileNotFoundError, OSError, pickle.PickleError):
    model = None
    scaler = None


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_form_data(form_data):
    errors = {}
    gender = str(form_data.get("personGender", "")).strip().lower()

    if not gender or gender not in {"male", "female"}:
        errors["personGender"] = "Please select a valid gender."

    for field_name, (min_value, max_value) in REFERENCE_RANGES.items():
        raw_value = form_data.get(field_name)
        if raw_value is None or str(raw_value).strip() == "":
            errors[field_name] = "This field is required."
            continue

        value = parse_float(raw_value)
        if value is None:
            errors[field_name] = "Please enter a valid number."
            continue

        if not (min_value <= value <= max_value):
            errors[field_name] = f"Value must be between {min_value} and {max_value}."

    for field_name, range_values in GENDER_SPECIFIC_RANGES.get(gender, {}).items():
        raw_value = form_data.get(field_name)
        if raw_value is None or str(raw_value).strip() == "":
            errors[field_name] = "This field is required."
            continue

        value = parse_float(raw_value)
        if value is None:
            errors[field_name] = "Please enter a valid number."
            continue

        min_value, max_value = range_values
        if not (min_value <= value <= max_value):
            errors[field_name] = f"Value must be between {min_value} and {max_value} for {gender.title()} gender."

    return errors


@app.route("/")
def index():
    return render_template("index.html", errors={}, form={})


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    errors = validate_form_data(form_data)

    if errors:
        return render_template("index.html", errors=errors, form=form_data)

    if model is None:
        return render_template(
            "result.html",
            message="Model files are missing. Please add the trained model and scaler files in the models folder.",
            form=form_data,
            prediction_value="N/A",
        )

    if scaler is None:
            return render_template(
                "result.html",
                message="scalar files are missing. Please add the trained model and scaler files in the models folder.",
                form=form_data,
                prediction_value="N/A",
            )

    person_age = float(form_data.get("personAge"))
    person_gender_raw = form_data.get("personGender")
    person_gender = 0 if person_gender_raw == "Male" else 1
    blood_glucose_random = float(form_data.get("blood_glucose_random"))
    blood_urea = float(form_data.get("blood_urea"))
    serum_creatinine = float(form_data.get("serum_creatinine"))
    sodium = float(form_data.get("sodium"))
    potassium = float(form_data.get("potassium"))
    hemoglobin = float(form_data.get("hemoglobin"))
    packed_cell_volume = float(form_data.get("packed_cell_volume"))
    white_blood_cell = float(form_data.get("white_blood_cell"))
    red_blood_cell = float(form_data.get("red_blood_cell"))

    input_features = [
        person_age,        
        blood_glucose_random,
        blood_urea,
        serum_creatinine,
        sodium,
        potassium,
        hemoglobin,
        packed_cell_volume,
        white_blood_cell,
        red_blood_cell,
        person_gender
    ]

    preprocessed_input = scaler.transform([input_features])
    prediction_result = model.predict(preprocessed_input)
    prediction_value = int(prediction_result[0])
    prediction_label = "CKD Detected" if prediction_value == 1 else "No CKD Detected"

    print("\n=== PREDICTION INPUT ===")
    print(f"personAge: {person_age}")
    print(f"personGender: {person_gender_raw}")
    print(f"blood_glucose_random: {blood_glucose_random}")
    print(f"blood_urea: {blood_urea}")
    print(f"serum_creatinine: {serum_creatinine}")
    print(f"sodium: {sodium}")
    print(f"potassium: {potassium}")
    print(f"hemoglobin: {hemoglobin}")
    print(f"packed_cell_volume: {packed_cell_volume}")
    print(f"white_blood_cell: {white_blood_cell}")
    print(f"red_blood_cell: {red_blood_cell}")
    print(f"person_gender_numeric: {person_gender}")
    print(f"preprocessed_input: {preprocessed_input}")
    print(f"prediction_result: {prediction_result}")
    print(f"prediction_label: {prediction_label}")
    print("=== END PREDICTION ===\n")

    return render_template(
        "result.html",
        message=prediction_label,
        form=form_data,
        prediction_value=prediction_value,
    )


if __name__ == "__main__":
    app.run(debug=True)