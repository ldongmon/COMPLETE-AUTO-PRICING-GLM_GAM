"""
Unit tests for src/utils/hand_calc.py: the term-by-term table must rebuild
the GLM prediction exactly, and the dataset view must highlight the right rows.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.lightweight_glm import LightweightGLM
from src.utils.hand_calc import glm_terms_table, show_policy_in_dataset, where_value_comes_from


def test_terms_table_rebuilds_the_prediction():
    # A GLM with 3 columns; the policy has VehBrand_B2 = 0, so that line must not appear.
    glm = LightweightGLM(pd.Series({"const": -3.0, "BonusMalus": 0.02, "log_Density": 0.1, "VehBrand_B2": 0.5}))
    raw = pd.Series({"IDpol": 7, "BonusMalus": 60, "Density": 100, "VehBrand": "B1"})
    encoded = pd.Series({"BonusMalus": 60.0, "log_Density": np.log(100), "VehBrand_B2": 0.0})
    table = glm_terms_table(glm, encoded, raw)
    assert list(table["term"]) == ["const", "BonusMalus", "log_Density"]
    eta = -3.0 + 0.02 * 60 + 0.1 * np.log(100)  # by hand
    assert table["b x x"].sum() == pytest.approx(eta)
    X = pd.DataFrame([{"const": 1.0, **encoded.to_dict()}])
    assert np.exp(table["b x x"].sum()) == pytest.approx(glm.predict(X)[0])


def test_where_value_comes_from_names_the_raw_column():
    raw = pd.Series({"Density": 1185, "DrivAge": 30, "VehAge": 3, "VehBrand": "B12", "BonusMalus": 95})
    assert where_value_comes_from("log_Density", raw) == "ln(Density = 1,185)"
    assert where_value_comes_from("DrivAge_bin_25-34", raw) == "DrivAge = 30 is in band 25-34 -> 1"
    assert where_value_comes_from("VehBrand_B12", raw) == "VehBrand = B12 -> 1"
    assert where_value_comes_from("BonusMalus", raw) == "BonusMalus = 95"


def test_show_policy_in_dataset_keeps_neighbours_and_marks_the_policy():
    df = pd.DataFrame({"IDpol": [10, 11, 12, 13, 14, 15], "Exposure": [1.0, 0.5, 0.2, 1.0, 0.3, 0.9]})
    view = show_policy_in_dataset(df, [12], ["Exposure"], context=1)
    assert list(view.data["IDpol"]) == [11, 12, 13]  # the policy and one row on each side
    assert "fff3a0" in view.to_html()  # the highlight colour is applied
