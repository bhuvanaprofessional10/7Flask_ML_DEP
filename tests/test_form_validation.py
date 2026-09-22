import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import validate_form_data


def test_valid_form_values_pass():
    payload = {
        "personAge": "42",
        "personGender": "Male",
        "blood_glucose_random": "120",
        "blood_urea": "15",
        "serum_creatinine": "1.0",
        "sodium": "140",
        "potassium": "4.2",
        "hemoglobin": "14.5",
        "packed_cell_volume": "45",
        "white_blood_cell": "7000",
        "red_blood_cell": "5.0",
    }

    errors = validate_form_data(payload)
    assert errors == {}


def test_out_of_range_values_fail():
    payload = {
        "personAge": "42",
        "personGender": "Male",
        "blood_glucose_random": "500",
        "blood_urea": "6",
        "serum_creatinine": "1.5",
        "sodium": "130",
        "potassium": "3.0",
        "hemoglobin": "12.0",
        "packed_cell_volume": "35",
        "white_blood_cell": "12000",
        "red_blood_cell": "3.5",
    }

    errors = validate_form_data(payload)
    assert "blood_glucose_random" in errors
    assert "blood_urea" in errors
    assert "serum_creatinine" in errors
    assert "sodium" in errors
    assert "potassium" in errors
    assert "packed_cell_volume" in errors
    assert "white_blood_cell" in errors
    assert "red_blood_cell" in errors
