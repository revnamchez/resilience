"""
unemp.py - Streamlit GUI for the PhD labour-market ML study.

Run:
    streamlit run unemp.py

Put occupazione.csv and disoccupazione.csv beside this file, or upload
both files in the sidebar.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

st.set_page_config(page_title="West Africa Labour-Market ML System",
                   page_icon="📊", layout="wide")

COUNTRIES = [
    "Benin","Burkina Faso","Cabo Verde","Côte d'Ivoire","Gambia","Ghana",
    "Guinea","Guinea-Bissau","Liberia","Mali","Mauritania","Niger",
    "Nigeria","Senegal","Sierra Leone","Togo"
]
FEATURES = ["year","employment","baseline_2020","recovery_score","sex","age"]
NUMERIC = ["year","employment","baseline_2020","recovery_score"]
CATEGORICAL = ["sex","age"]
DATA_DIR = Path(__file__).resolve().parent

def prepare_data(emp_file, unemp_file):
    emp = pd.read_csv(emp_file)
    unemp = pd.read_csv(unemp_file)
    if "obs_value" in emp: emp = emp.rename(columns={"obs_value":"employment"})
    if "obs_value" in unemp: unemp = unemp.rename(columns={"obs_value":"unemployment"})
    keys = ["iso_code","country","sex","age","year"]
    req_e = set(keys + ["employment"])
    req_u = set(keys + ["unemployment"])
    if not req_e.issubset(emp.columns):
        raise ValueError(f"Employment file is missing: {sorted(req_e-set(emp.columns))}")
    if not req_u.issubset(unemp.columns):
        raise ValueError(f"Unemployment file is missing: {sorted(req_u-set(unemp.columns))}")
    df = emp[keys+["employment"]].merge(
        unemp[keys+["unemployment"]], on=keys, how="inner", validate="one_to_one"
    )
    df = df[df["country"].isin(COUNTRIES) & df["year"].between(2020,2025)].copy()
    df = df.sort_values(["country","sex","age","year"]).reset_index(drop=True)
    base = (df[df["year"]==2020][["country","sex","age","employment"]]
            .rename(columns={"employment":"baseline_2020"}))
    df = df.merge(base, on=["country","sex","age"], how="left",
                  validate="many_to_one")
    df["recovery_score"] = df["employment"] - df["baseline_2020"]
    return df.dropna(subset=["employment","unemployment","baseline_2020"]).copy()

def preprocessor():
    num = Pipeline([("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())])
    cat = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("num",num,NUMERIC),("cat",cat,CATEGORICAL)])

def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=500,max_depth=4,learning_rate=0.05,
                                subsample=0.9,colsample_bytree=0.9,
                                objective="reg:squarederror",random_state=42,n_jobs=-1),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64,32,16),max_iter=1500,
                                       early_stopping=True,validation_fraction=0.15,
                                       alpha=0.0005,random_state=42)
    }

@st.cache_resource
def fit_models(df):
    models = {}
    X, y = df[FEATURES], df["unemployment"]
    for name, estimator in build_models().items():
        models[name] = Pipeline([("preprocess",preprocessor()),("model",estimator)])
        models[name].fit(X,y)

    cutoff = float(y.median())
    clf = Pipeline([
        ("preprocess",preprocessor()),
        ("model",XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,
                               subsample=0.9,colsample_bytree=0.9,
                               objective="binary:logistic",eval_metric="logloss",
                               random_state=42,n_jobs=-1))
    ])
    clf.fit(X,(y>cutoff).astype(int))
    return models, clf, cutoff

@st.cache_data
def load_uploaded(emp_bytes,unemp_bytes):
    from io import BytesIO
    return prepare_data(BytesIO(emp_bytes),BytesIO(unemp_bytes))

# ----------------------- LOAD DATA -----------------------
st.title("West Africa Labour-Market Decision-Support Framework")
st.caption("Explainable and Robust Machine Learning for Labour-Market Resilience and Unemployment Vulnerability Prediction")

st.markdown("---")
st.markdown("**PhD Research by: Cynthia Ifunanya Udoaku.**")
st.markdown("Designed by: Nnaemeka U. Ezeonyi, PhD.")


st.sidebar.header("Data")
ue = st.sidebar.file_uploader("Upload employment CSV",type=["csv"])
uu = st.sidebar.file_uploader("Upload unemployment CSV",type=["csv"])

try:
    if ue and uu:
        W_A = load_uploaded(ue.getvalue(),uu.getvalue())
        source = "Uploaded CSV files"
    else:
        ep, up = DATA_DIR/"occupazione.csv", DATA_DIR/"disoccupazione.csv"
        if not (ep.exists() and up.exists()):
            st.warning("Upload both CSV files, or place occupazione.csv and disoccupazione.csv beside unemp.py.")
            st.stop()
        W_A = prepare_data(ep,up)
        source = "Local CSV files"
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

models, classifier, cutoff = fit_models(W_A)

st.sidebar.success(source)
st.sidebar.write(f"Observations: **{len(W_A):,}**")
st.sidebar.write(f"Countries: **{W_A.country.nunique()}**")
st.sidebar.write(f"Period: **{int(W_A.year.min())}-{int(W_A.year.max())}**")

# ----------------------- INPUT -----------------------
with st.sidebar:
    st.header("Prediction Input")
    country = st.selectbox("Country",COUNTRIES)
    sex = st.selectbox("Sex",sorted(W_A.sex.unique()))
    age = st.selectbox("Age group",sorted(W_A.age.unique()))
    year = st.number_input("Year",2020,2035,2025,1)
    employment = st.number_input("Employment",0.0,float(W_A.employment.median()),0.1)
    baseline = st.number_input("2020 baseline employment",0.0,float(W_A.baseline_2020.median()),0.1)
    model_name = st.selectbox("Regression model",list(models),index=2)

recovery = employment - baseline
input_df = pd.DataFrame([{
    "year":year,"employment":employment,"baseline_2020":baseline,
    "recovery_score":recovery,"sex":sex,"age":age
}])

tab1,tab2,tab3,tab4 = st.tabs(["Prediction","Vulnerability","Country Analysis","Explainability"])

with tab1:
    st.header("Unemployment Prediction")
    c1,c2,c3 = st.columns(3)
    c1.metric("Country",country)
    c2.metric("Recovery Score",f"{recovery:.3f}")
    if st.button("Predict unemployment",type="primary"):
        pred=float(models[model_name].predict(input_df)[0])
        c3.metric("Predicted unemployment",f"{pred:.3f}")
        st.info("Country is used for context and reporting. It is not a direct model feature.")
        st.caption("Predictions for a completely unseen country should be treated cautiously because the research found limited cross-country generalisation.")

with tab2:
    st.header("Unemployment Vulnerability")
    st.metric("Reference cutoff",f"{cutoff:.3f}")
    if st.button("Assess vulnerability",type="primary"):
        prob=float(classifier.predict_proba(input_df)[0,1])
        threshold=0.50
        classification = "High vulnerability" if prob >= threshold else "Not high vulnerability"

        st.metric("Vulnerability probability",f"{prob:.3f}")
        st.metric("Classification",classification)

        if prob >= threshold:
            st.warning(
                f"The model classifies this observation as high vulnerability because "
                f"the probability ({prob:.3f}) is at or above the 0.500 threshold."
            )
        else:
            st.success(
                f"The model does not classify this observation as high vulnerability because "
                f"the probability ({prob:.3f}) is below the 0.500 threshold."
            )
        st.caption("A probability below 0.500 is not classified as high vulnerability. This is a model prediction, not a causal judgement.")

with tab3:
    st.header("Country Analysis")
    W_A["vulnerability_probability"] = classifier.predict_proba(W_A[FEATURES])[:,1]
    summary=(W_A.groupby("country")
             .agg(mean_recovery=("recovery_score","mean"),
                  mean_unemployment=("unemployment","mean"),
                  mean_vulnerability=("vulnerability_probability","mean"))
             .reset_index()
             .sort_values("mean_vulnerability",ascending=False))
    st.dataframe(summary.style.format({
        "mean_recovery":"{:.3f}","mean_unemployment":"{:.3f}",
        "mean_vulnerability":"{:.3f}"
    }),use_container_width=True,hide_index=True)

    fig,ax=plt.subplots(figsize=(10,7))
    p=summary.sort_values("mean_vulnerability")
    ax.barh(p.country,p.mean_vulnerability)
    ax.set_xlim(0,1)
    ax.set_xlabel("Mean Vulnerability Probability")
    ax.set_ylabel("Country")
    ax.set_title("Country Vulnerability Ranking")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    corr=summary.mean_recovery.corr(summary.mean_vulnerability)
    st.write(f"Country-level recovery-vulnerability correlation: **{corr:.3f}**")

with tab4:
    st.header("Explainability")
    st.write("The following outputs describe model behaviour; they do not establish causation.")
    xgb=models["XGBoost"]
    names=xgb.named_steps["preprocess"].get_feature_names_out()
    imp=xgb.named_steps["model"].feature_importances_
    imp_df=pd.DataFrame({"feature":names,"importance":imp}).sort_values("importance",ascending=False)
    st.subheader("XGBoost Feature Importance")
    st.dataframe(imp_df.style.format({"importance":"{:.4f}"}),use_container_width=True,hide_index=True)

    fig,ax=plt.subplots(figsize=(10,7))
    p=imp_df.head(10).sort_values("importance")
    ax.barh(p.feature,p.importance)
    ax.set_xlabel("Importance")
    ax.set_title("Top XGBoost Features")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    try:
        perm=permutation_importance(xgb,W_A[FEATURES],W_A["unemployment"],
                                    scoring="r2",n_repeats=10,random_state=42,n_jobs=-1)
        pdf=pd.DataFrame({"feature":FEATURES,
                          "importance_mean":perm.importances_mean,
                          "importance_std":perm.importances_std}).sort_values("importance_mean",ascending=False)
        st.subheader("Permutation Importance")
        st.dataframe(pdf.style.format({"importance_mean":"{:.4f}","importance_std":"{:.4f}"}),
                     use_container_width=True,hide_index=True)
    except Exception as e:
        st.warning(f"Permutation importance unavailable: {e}")

st.divider()
st.caption("PhD research prototype. Decision-support only; human judgement remains important.")
