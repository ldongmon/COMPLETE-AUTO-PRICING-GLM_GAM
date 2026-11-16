"""
hand_calc.py
============
Shows, for ONE policy of the dataset, how a GLM prediction is built, term by
term, and where every number comes from:

    term           the model column (const, BonusMalus, log_Density, VehBrand_B12, ...)
    x              the value this policy has in that column
    where x comes from   the raw column of the dataset it is computed from
    b              the coefficient fitted by the GLM (read from the saved model)
    b x x          the contribution to the linear predictor eta
    exp(b x x)     the same contribution, as a multiplier

    eta = sum of the "b x x" column        prediction = exp(eta)

Used by notebooks 02 to 11 and by the app, so every hand calculation reads the same way.
show_policy_in_dataset() shows the policy inside the dataset, highlighted, like a screenshot.
"""

import numpy as np  # maths
import pandas as pd  # tables


def show_policy_in_dataset(df, idpols, columns, context=2):
    """
    A view of the dataset around the chosen policies, like a screenshot of the
    file: `context` rows before and after each one, the chosen rows highlighted
    in yellow. The left-hand number is the row's position in the table.
    """
    positions = np.flatnonzero(df["IDpol"].isin(list(idpols)))  # row numbers of the chosen policies
    keep = sorted({p + k for p in positions for k in range(-context, context + 1) if 0 <= p + k < len(df)})  # with their neighbours
    view = df.iloc[keep][["IDpol"] + [c for c in columns if c != "IDpol"]].copy()  # the extract
    view.index = pd.Index(keep, name="row")  # position in the table
    chosen = set(idpols)

    def highlight(row):  # yellow background for the chosen policies
        colour = "background-color: #fff3a0; font-weight: bold" if row["IDpol"] in chosen else ""
        return [colour] * len(row)

    return view.style.apply(highlight, axis=1).format(precision=4)


def steps_table(rows):
    """
    A hand calculation as a table, one line per step:
        step | calculation (with the numbers) | result | source (dataset column, coefficient or assumption)
    `rows` is a list of (step, calculation, result, source) tuples.
    """
    table = pd.DataFrame(rows, columns=["step", "calculation", "result", "source"])
    return table.style.hide(axis="index").set_properties(**{"text-align": "left"})


def where_value_comes_from(column, raw):
    """Plain-language origin of the encoded value `column`, for the raw policy row `raw`."""
    if column == "const":
        return "intercept: always 1"
    if column == "log_Density":
        return f"ln(Density = {raw['Density']:,.0f})"
    if column == "VehAge_capped":
        return f"VehAge = {raw['VehAge']} (capped at 24)"
    if column.startswith("DrivAge_bin_"):
        return f"DrivAge = {raw['DrivAge']} is in band {column[len('DrivAge_bin_'):]} -> 1"
    if column.startswith("VehAge_bin_"):
        return f"VehAge = {raw['VehAge']} is in band {column[len('VehAge_bin_'):]} -> 1"
    for factor in ("VehBrand", "VehGas", "Region", "Area"):  # category columns: one 0/1 column per level
        if column.startswith(factor + "_"):
            return f"{factor} = {raw[factor]} -> 1"
    if column in raw.index:  # numeric column used as it is
        return f"{column} = {raw[column]}"
    return "computed column"


def glm_terms_table(results, encoded_row, raw_row):
    """
    One line per term that is "on" for this policy (x different from 0).
      results      fitted statsmodels GLM (gives the coefficients b)
      encoded_row  the policy's encoded columns (one row of the encoder's output)
      raw_row      the same policy's raw columns, as in the dataset
    """
    rows = [{"term": "const", "x": 1.0, "where x comes from": where_value_comes_from("const", raw_row),
             "b (fitted coefficient)": results.params["const"]}]
    for column in results.model.exog_names[1:]:  # every column of the model
        value = float(encoded_row[column])
        if value != 0:  # a 0 adds nothing (a level the policy does not have, a reference band)
            rows.append({"term": column, "x": value, "where x comes from": where_value_comes_from(column, raw_row),
                         "b (fitted coefficient)": results.params[column]})
    table = pd.DataFrame(rows)
    table["b x x"] = table["b (fitted coefficient)"] * table["x"]  # contribution to eta
    table["exp(b x x)"] = np.exp(table["b x x"])  # the same, as a multiplier
    return table
