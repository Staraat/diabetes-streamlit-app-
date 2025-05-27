# ~~~  Diabetes Risk Streamlit App  by Cheah Jun Hong [TP081634] ~~~
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit.components.v1 as components

# Helper to embed SHAP plots
shap.initjs()
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 300, scrolling=True)

# Path setup
BASE = Path(__file__).parent
model = joblib.load(BASE / 'model7_xgb_interactions.pkl')
scaler = joblib.load(BASE / 'scaler.pkl')
with open(BASE / 'feature_list.json') as f:
    feature_list = json.load(f)
scaled_features = ["BMI", "MentHlth", "PhysHlth"]
explainer = shap.TreeExplainer(model)

# Friendly feature names and recommendations
friendly_names = {
    "BMI": "Body‑Mass‑Index",
    "Age": "Age",
    "GenHlth": "General health",
    "MentHlth": "Mental‑health days",
    "PhysHlth": "Physical‑health days",
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "HeartDiseaseorAttack": "Heart disease / heart attack",
    "Sex": "Gender",
    "BMI_Age": "BMI×Age",
    "HighBP_PhysHlth": "BP×Physical‑health",
    "DiffWalk_Age": "Walking diff × Age",
    "HeartAttack_PhysActivity": "Heart issue × Activity",
    "BMI_GenHlth": "BMI×General‑health",
}

recommendations = {
    "BMI": "Consider regular exercise and a balanced diet to reduce BMI.",
    "GenHlth": "Improve overall health through lifestyle changes (sleep, diet, activity).",
    "PhysHlth": "Consult a doctor about persistent physical discomfort.",
    "MentHlth": "Stress‑management or counselling may help mental well‑being.",
    "HighBP": "Manage blood‑pressure with diet, exercise or medication as prescribed.",
    "HighChol": "Reduce saturated‑fat intake to lower cholesterol.",
    "HeartDiseaseorAttack": "Follow cardiologist advice for heart‑condition management.",
    "BMI_GenHlth": "Working on weight and overall health will greatly lower risk.",
}

# Load GI & GL Excel 
gi_gl_df = pd.read_excel(BASE / 'GI & GL.xlsx')
gi_gl_df = gi_gl_df[gi_gl_df['GI'].notna() & (gi_gl_df['GI'] >= 0)]
gi_gl_df['Name'] = gi_gl_df['Name'].astype(str).str.strip()

def gi_category(gi):
    if gi <= 55: return 'Low'
    elif gi <= 69: return 'Medium'
    return 'High'

def gl_category(gl):
    if gl <= 10: return 'Low'
    elif gl <= 19: return 'Medium'
    return 'High'

gi_gl_df['GI Category'] = gi_gl_df['GI'].apply(gi_category)
gi_gl_df['GL Category'] = gi_gl_df['GL'].apply(gl_category)

# Page setup 
st.set_page_config(page_title="Diabetes Risk & GI Calculator", layout="wide")
if "show_privacy" not in st.session_state:
    st.session_state.show_privacy = True

if st.session_state.show_privacy:
    with st.container():
        st.warning("**Privacy Notice**  \n"
               "This website does **NOT** store or transmit any data you enter. "
               "All calculations run locally in your browser session.")
        if st.button("Understood and Agreed"):
            st.session_state.show_privacy = False

# Tabs for the app
tab1, tab2 = st.tabs(["🧠 Risk Predictor", "🥗 GI & GL Calculator"])

with tab1:
    st.title("🩺 Diabetes Risk Prediction")
    with st.form("predict_form", clear_on_submit=False):
        st.subheader("👤 Demographic Information")
        col1, col2 = st.columns(2)
        with col1:
            sex = st.radio("Gender", options=[0,1], format_func=lambda x: "Female" if x==0 else "Male")
            age_code = st.selectbox("Age group", list(range(1,14)), format_func=lambda x: ["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80–99"][x-1])
        with col2:
            education = st.selectbox("Education level", [1,2,3,4,5,6], format_func=lambda x: ["Never/Kindergarten","Grades 1–8","Grades 9–11","High‑school graduate","College 1–3 yrs","College 4+ yrs"][x-1])
            income = st.selectbox("Household income", list(range(1,12)), format_func=lambda x: ["< $10k","$10k–<15k","$15k–<20k","$20k–<25k","$25k–<35k","$35k–<50k","$50k–<75k","$75k–<100k","$100k–<150k","$150k–<200k","≥ $200k"][x-1])

        st.subheader("🏃 Lifestyle & Habits")
        col1, col2, col3 = st.columns(3)
        yn = lambda: [0,1]; fmt = lambda x: "No" if x==0 else "Yes"
        with col1:
            phys_act = st.selectbox("Physically active?", yn(), format_func=fmt)
            fruits    = st.selectbox("Eat fruit daily?", yn(), format_func=fmt)
        with col2:
            veggies   = st.selectbox("Eat vegetables daily?", yn(), format_func=fmt)
            smoker    = st.selectbox("Current smoker?", yn(), format_func=fmt)
        with col3:
            alcohol   = st.selectbox("Heavy alcohol use?", yn(), format_func=fmt)
            any_hc    = st.selectbox("Any healthcare cover?", yn(), format_func=fmt)

        st.subheader("🩺 Medical History")
        col1, col2, col3 = st.columns(3)
        with col1:
            high_bp   = st.selectbox("Diagnosed high blood‑pressure?", yn(), format_func=fmt)
            high_chol = st.selectbox("Diagnosed high cholesterol?", yn(), format_func=fmt)
        with col2:
            chol_check= st.selectbox("Cholesterol check (last 5 yrs)?", yn(), format_func=fmt)
            heart_dis = st.selectbox("Heart disease / attack history?", yn(), format_func=fmt)
        with col3:
            stroke    = st.selectbox("Ever had a stroke?", yn(), format_func=fmt)
            diffwalk  = st.selectbox("Difficulty walking?", yn(), format_func=fmt)
            nodoccost = st.selectbox("Skipped doctor due to cost?", yn(), format_func=fmt)

        st.subheader("📋 Current Health Status")
        use_bmi_calc = st.checkbox("I don't know my BMI, help me calculate it")

        if use_bmi_calc:
            height_cm = st.number_input("Enter your height (cm):", min_value=100.0, max_value=250.0, value=170.0)
            weight_kg = st.number_input("Enter your weight (kg):", min_value=30.0, max_value=200.0, value=65.0)
        
            height_m = height_cm / 100
            bmi_value = round(weight_kg / (height_m ** 2), 2)
        
            st.success(f"✅ Your calculated BMI is: **{bmi_value} kg/m²**")
            bmi = st.slider("Body-Mass-Index (kg/m²)", 10.0, 50.0, value=bmi_value)
        else:
            bmi = st.slider("Body-Mass-Index (kg/m²)", 10.0, 50.0, value=25.0)

        gen  = st.selectbox("General health rating", [1,2,3,4,5], format_func=lambda x:["Excellent","Very good","Good","Fair","Poor"][x-1])
        col_mh, col_ph = st.columns(2)
        with col_mh:
            ment = st.slider("Days mental health not good (0‑30)",0,30,0)
        with col_ph:
            phys = st.slider("Days physical health not good (0‑30)",0,30,0)

        user_input = dict(
            HighBP=high_bp, HighChol=high_chol, CholCheck=chol_check, BMI=bmi,
            Smoker=smoker, Stroke=stroke, HeartDiseaseorAttack=heart_dis,
            PhysActivity=phys_act, Fruits=fruits, Veggies=veggies,
            HvyAlcoholConsump=alcohol, AnyHealthcare=any_hc, NoDocbcCost=nodoccost,
            GenHlth=gen, MentHlth=ment, PhysHlth=phys, DiffWalk=diffwalk,
            Sex=sex, Age=age_code, Education=education, Income=income
        )

        predict_btn = st.form_submit_button("Predict")
        clear_btn   = st.form_submit_button("Clear")

    if clear_btn:
        st.session_state.clear()
        st.session_state.show_privacy = True
        st.stop()

    if predict_btn:
        user_input["BMI_Age"] = user_input["BMI"] * user_input["Age"]
        user_input["HighBP_PhysHlth"] = user_input["HighBP"] * user_input["PhysHlth"]
        user_input["DiffWalk_Age"] = user_input["DiffWalk"] * user_input["Age"]
        user_input["HeartAttack_PhysActivity"] = user_input["HeartDiseaseorAttack"] * user_input["PhysActivity"]
        user_input["BMI_GenHlth"] = user_input["BMI"] * user_input["GenHlth"]

        user_df = pd.DataFrame([user_input])[feature_list]
        user_df_scaled = user_df.copy()
        user_df_scaled[scaled_features] = scaler.transform(user_df[scaled_features])

        proba = model.predict_proba(user_df_scaled)[0][1]
        prediction = "High Risk" if proba >= 0.5 else "Low Risk"
        st.success(f"🧠 Prediction: {prediction} ({proba:.2%} probability)")

        shap_vals = explainer.shap_values(user_df_scaled)
        shap_plot = shap.force_plot(explainer.expected_value, shap_vals[0], user_df.iloc[0], matplotlib=False)
        st_shap(shap_plot, height=200)

        shap_df = pd.DataFrame({"Feature": user_df.columns, "SHAP": shap_vals[0], "Value": user_df.iloc[0].values})
        pos = shap_df[shap_df.SHAP>0].nlargest(3,"SHAP")
        neg = shap_df[shap_df.SHAP<0].nsmallest(3,"SHAP")

        inc_txt = ", ".join(friendly_names.get(f,f) for f in pos.Feature)
        red_txt = ", ".join(friendly_names.get(f,f) for f in neg.Feature)
        st.subheader("🗣️ Explanation Summary")
        expl = f"Risk increased by: {inc_txt}." + (f" Risk reduced by: {red_txt}." if red_txt else "")
        st.write(expl)

        st.subheader("📌 Personalised Recommendations")
        rec_keys = set(pos.Feature) | set(neg.Feature)
        for k in rec_keys:
            if k in recommendations:
                st.write("•", recommendations[k])

    st.markdown("""
    ---
    ##### ❗ Disclaimer
    *This tool only provides statistical risk estimates and may be incorrect. It does **NOT** diagnose or treat medical conditions.
    Please consult a licensed healthcare provider for accurate medical advice and decisions.*
    """)

with tab2:
    st.title("🥗 Glycemic Index & Load Calculator")
    food_list = sorted(gi_gl_df['Name'].unique())
    food = st.selectbox("Select a food item:", food_list)

    # start with empty meal basket
    if 'meal_basket' not in st.session_state:
        st.session_state.meal_basket = []

    if food:
        row = gi_gl_df[gi_gl_df['Name'] == food].iloc[0]
        gi = row['GI']
        gl = row['GL']
        cat_gi = gi_category(gi)
        cat_gl = gl_category(gl)

        color_gi = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(cat_gi, "⚪")
        color_gl = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(cat_gl, "⚪")

        st.markdown(f"**GI**: {gi} ({color_gi} {cat_gi})")
        st.markdown(f"**GL**: {gl} ({color_gl} {cat_gl})")

        if st.button("➕ Add to Meal"):
            st.session_state.meal_basket.append({"Name": food, "GI": gi, "GL": gl})
            st.success(f"Added {food} to your meal basket!")

    st.caption("*GI = Glycemic Index; GL = Glycemic Load per standard portion.*")

    # Display meal basket when there are something inside
    if st.session_state.meal_basket:
        st.markdown("### 🧺 Your Meal Basket")
        meal_df = pd.DataFrame(st.session_state.meal_basket)
        avg_gi = meal_df['GI'].mean()
        total_gl = meal_df['GL'].sum()
        st.dataframe(meal_df)
        st.markdown(f"**Average GI**: {avg_gi:.1f}")
        st.markdown(f"**Total GL**: {total_gl:.1f}")

        if st.button("🗑️ Clear Meal Basket"):
            st.session_state.meal_basket.clear()
    
    st.subheader("📌 Recommendations Based on Your Meal")
    
    if st.session_state.meal_basket:
        # Recompute summary
        meal_df = pd.DataFrame(st.session_state.meal_basket)
        total_gl = meal_df['GL'].sum()
        
        # Basic threshold suggestion
        if total_gl >= 20:
            st.write("• Your meal has a **high glycemic load**. Consider reducing portions or swapping for lower-GL items.")
        elif total_gl >= 11:
            st.write("• Your meal has a **moderate glycemic load**. Try balancing with more fiber or protein.")
        else:
            st.write("• Your meal has a **low glycemic load**. Well done!")

        # Smart pattern-based suggestions
        smart_suggestions = []

        patterns = {
            "white rice": "Replace **white rice** with **quinoa**, **barley**, or **brown rice**.",
            "white bread": "Use **whole-grain bread** instead of white bread.",
            "cornflakes": "Switch **cornflakes** for **steel-cut oats** or **unsweetened muesli**.",
            "sugar": "Limit items with **added sugar**; use natural sweeteners like fruit.",
            "syrup": "Avoid **syrupy toppings**; choose fresh fruit or yogurt.",
            "soft drink": "Swap soft drinks for **sparkling water** or **unsweetened iced tea**.",
            "soda": "Avoid soda; choose **infused water** or lemon water.",
            "honey": "Use **honey** sparingly — it still spikes blood sugar.",
            "instant noodles": "Try **soba**, **vermicelli**, or whole-wheat noodles.",
            "potato": "Boiled or roasted **sweet potatoes** have a lower GI than baked potatoes.",
            "fries": "Fries are high GL — consider **baked alternatives** or air-frying.",
            "juice": "Limit **fruit juice** — eat the whole fruit instead.",
            "cookies": "Swap cookies for **nuts**, **seeds**, or dark chocolate (in moderation).",
        }

        # Loop through items in the basket
        for item in st.session_state.meal_basket:
            item_name = item['Name'].lower()
            for key in patterns:
                if key in item_name:
                    smart_suggestions.append(patterns[key])

        # Remove duplicates
        smart_suggestions = list(dict.fromkeys(smart_suggestions))

        # Display
        if smart_suggestions:
            st.markdown("**Consider these improvements:**")
            for suggestion in smart_suggestions:
                st.write(f"• {suggestion}")
    else:
        st.info("🧺 Your meal basket is empty. Add items above to see GI/GL and get recommendations.")

    st.markdown("""
                ---
                ##### ❗ Disclaimer
                *The GI and GL values are from online resource, actual impact may vary.*""" \
                " *GI and GL are not the only factors to consider when making food choices.* ")
    st.markdown("*For more details, please refer to the [GI & GL Guide](https://glycemic-index.net/).*"
                " *For Glycemic Index related research and news, please refer to [Glycemic Index Research and GI News](https://glycemicindex.com/).*")
