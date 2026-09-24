from pathlib import Path
import pickle
import logging

from flask import Flask, render_template, request

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
	app.logger.addHandler(logging.StreamHandler())
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "school_fee_predict_MultiLinear_Final.sav"
BOARDS = ("CBSE", "IB", "ICSE", "IGCSE", "State Board")
BOARD_FEATURES = (
	"Board_Affiliation_CBSE",
	"Board_Affiliation_IB",
	"Board_Affiliation_ICSE",
	"Board_Affiliation_IGCSE",
	"Board_Affiliation_StateBoard",
)
CITY_TIERS = ("Tier 1", "Tier 2", "Tier 3")
CITY_FEATURES = (
	"City_Tier_Tier1",
	"City_Tier_Tier2",
	"City_Tier_Tier3",
)
FACILITY_FIELDS = ("smart_classrooms", "sports_facilities", "lab_facilities")


def load_model():
	if not MODEL_PATH.exists():
		return None
	try:
		with open("models/school_fee_predict_MultiLinear_Final.sav", "rb") as f:
			model = pickle.load(f)
			app.logger.info(f"Model loaded successfully ")
			return model
	except (OSError, pickle.PickleError, EOFError):
		app.logger.error(f"Error occurred while loading model from {MODEL_PATH}")
		return None


model = load_model()


def encode_board_affiliation(board):
	"""Return one-hot board values in the model's training-column order."""
	board_key = board.replace(" ", "")
	return [
		1 if feature == f"Board_Affiliation_{board_key}" else 0
		for feature in BOARD_FEATURES
	]


def encode_city_tier(city_tier):
	"""Return one-hot city-tier values in the model's training-column order."""
	tier_key = city_tier.replace(" ", "")
	return [
		1 if feature == f"City_Tier_{tier_key}" else 0
		for feature in CITY_FEATURES
	]


def validate_form(form):
	errors = {}
	grade = form.get("grade", "").strip()
	if not grade.isdigit() or not 1 <= int(grade) <= 12:
		errors["grade"] = "Choose a grade from 1 to 12."
	if form.get("board_affiliation") not in BOARDS:
		errors["board_affiliation"] = "Choose a valid board affiliation."
	if form.get("city_tier") not in CITY_TIERS:
		errors["city_tier"] = "Choose a valid city tier."
	for field in FACILITY_FIELDS:
		if form.get(field) not in {"Yes", "No"}:
			errors[field] = "Select Yes or No."
	return errors


def predict_fee(form):
	app.logger.info('This is an info message!') 
	ui_grade=int(form["grade"])
	ui_board=form["board_affiliation"]
	ui_tier=form["city_tier"]
	board_values = encode_board_affiliation(ui_board)
	city_values = encode_city_tier(ui_tier)
	facility_values = [1 if form[field] == "Yes" else 0  for field in FACILITY_FIELDS]
	app.logger.info(f'User Input - Grade: {ui_grade}, Board: {ui_board}, City Tier: {ui_tier}')
	app.logger.info(f'Board one-hot values: {board_values}')
	app.logger.info(f'City tier one-hot values: {city_values}')
	app.logger.info(f'Facility values - Smart classrooms, Sports facilities, Lab facilities: {facility_values}')
	features = [int(form["grade"])] + board_values + city_values +facility_values
	app.logger.info(f'Features: {features}')
	if model is not None:
		try:
			app.logger.info(f'Predicting with features: {features}')
			output = float(model.predict([features])[0])
			app.logger.info(f'Predicted output: {round(output, 2)}')
			return output
		except (AttributeError, TypeError, ValueError):
			app.logger.exception('Model prediction failed; using fallback estimate.')
	board_base = {"CBSE": 68000, "IB": 125000, "ICSE": 82000, "IGCSE": 105000, "State Board": 42000}
	tier_adjustment = {"Tier 1": 30000, "Tier 2": 12000, "Tier 3": 0}
	facilities = sum(facility_values) * 7500
	return board_base[form["board_affiliation"]] + tier_adjustment[form["city_tier"]] + facilities + (int(form["grade"]) - 1) * 1800


@app.route("/", methods=["GET", "POST"])
def index():
	form = request.form if request.method == "POST" else {}
	errors = validate_form(form) if request.method == "POST" else {}
	prediction = predict_fee(form) if request.method == "POST" and not errors else None
	return render_template("index.html", form=form, errors=errors, prediction=prediction)


if __name__ == "__main__":
	app.run(debug=True)
