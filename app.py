import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Safe Data Tool", layout="wide")
st.title("Safe Data Tool – Privacy Risk Assessment")

# ------------------- Helper functions -------------------
def add_laplace_noise(df, column, scale=1000, random_state=None):
    df = df.copy()
    if random_state is not None:
        np.random.seed(random_state)
    noise = np.random.laplace(loc=0, scale=scale, size=len(df))
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")
    df[column] = (df[column] + noise).round().astype(int)
    return df

def generalise_age(df, column="age"):
    df = df.copy()
    if column not in df.columns:
        return df
    try:
        bins = [0, 20, 30, 40, 50, 60, 70, 120]
        labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        df[column] = pd.cut(df[column], bins=bins, labels=labels, right=False)
    except Exception:
        pass
    return df

def calculate_risk(microdata, true_ids, quasi_columns):
    missing_in_micro = [c for c in quasi_columns if c not in microdata.columns]
    missing_in_true = [c for c in quasi_columns if c not in true_ids.columns]
    if missing_in_micro or missing_in_true:
        raise KeyError(f"Quasi-identifiers missing. In micro: {missing_in_micro}, in true_ids: {missing_in_true}")
    merged = microdata.merge(true_ids, on=quasi_columns, how="inner", suffixes=("_micro", "_true"))
    match_count = len(merged)
    risk_percent = (match_count / len(microdata)) * 100 if len(microdata) > 0 else 0.0
    return match_count, risk_percent

# ------------------- UI -------------------
st.write("Upload two CSV files — (1) microdata (dataset to anonymise) and (2) true identifiers (for testing risk).")

# File uploaders
micro_file = st.file_uploader("Upload Microdata CSV", type="csv")
true_file = st.file_uploader("Upload True Identifiers CSV (testing only)", type="csv")

# ✅ If files not uploaded, load from sample CSV in repo
if micro_file is None and os.path.exists("sample_microdata.csv"):
    micro_file = open("sample_microdata.csv", "rb")
if true_file is None and os.path.exists("sample_true_ids.csv"):
    true_file = open("sample_true_ids.csv", "rb")

# ✅ Stop if still no files found
if micro_file is None or true_file is None:
    st.error("No data files found. Please upload files or ensure sample CSV exists in the app directory.")
    st.stop()

# ✅ Read CSVs
try:
    micro_df = pd.read_csv(micro_file)
    true_df = pd.read_csv(true_file)
except Exception as e:
    st.error(f"Error reading CSV files: {e}")
    st.stop()

# ------------------- Show Data -------------------
st.subheader("📊 Data Preview")
st.write("Microdata sample:")
st.dataframe(micro_df.head())

st.write("True identifiers sample:")
st.dataframe(true_df.head())

# ------------------- Risk Assessment -------------------
common_cols = list(set(micro_df.columns).intersection(set(true_df.columns)))
default_quasi = [c for c in ["age", "gender", "district"] if c in common_cols]
if not default_quasi:
    default_quasi = common_cols[:3]

st.subheader("🔍 Step 1: Risk Assessment")
quasi_cols = st.multiselect("Select quasi-identifiers", options=common_cols, default=default_quasi)

if st.button("Calculate Risk"):
    try:
        matches, risk = calculate_risk(micro_df, true_df, quasi_cols)
        st.success(f"Matched Records: {matches} | Risk: {risk:.2f}%")
    except Exception as e:
        st.error(f"Error calculating risk: {e}")

# ------------------- Privacy Enhancement -------------------
st.subheader("🛡 Step 2: Privacy Enhancement")
numeric_cols = list(micro_df.select_dtypes(include=np.number).columns)

if not numeric_cols:
    st.warning("No numeric columns found in microdata.")
else:
    num_col = st.selectbox("Select numeric column to add noise", options=numeric_cols)
    noise_scale = st.slider("Noise scale (Laplace)", min_value=100, max_value=5000, value=1500, step=100)
    random_seed = st.number_input("Random seed (optional)", min_value=0, max_value=9999999, value=42, step=1)

    if st.button("Apply Privacy"):
        try:
            anon_df = add_laplace_noise(micro_df, num_col, scale=noise_scale, random_state=int(random_seed))
            if "age" in anon_df.columns:
                anon_df = generalise_age(anon_df, "age")
            st.write("Anonymised Data Sample:")
            st.dataframe(anon_df.head())

            fig, ax = plt.subplots()
            ax.hist(micro_df[num_col].dropna(), bins=30, alpha=0.6, label="Original")
            ax.hist(anon_df[num_col].dropna(), bins=30, alpha=0.6, label="Anonymised")
            ax.legend()
            st.pyplot(fig)

            new_quasi = [q for q in quasi_cols if q != "age"]
            matches_after, risk_after = calculate_risk(anon_df, true_df, new_quasi)
            st.info(f"After Enhancement → Matched Records: {matches_after} | Risk: {risk_after:.2f}%")

            csv = anon_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Anonymised CSV", data=csv, file_name="anonymised_data.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error during privacy enhancement: {e}")
