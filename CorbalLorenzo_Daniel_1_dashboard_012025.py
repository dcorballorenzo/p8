import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import logging
import base64
import matplotlib.pyplot as plt
import shap
from sklearn.cluster import KMeans

# ----- Load data and configure logging -----
current_directory = os.path.dirname(os.path.abspath(__file__))

path_df_train = os.path.join(current_directory, "data/dashboard/df_train.csv")
path_df_test = os.path.join(current_directory, "data/dashboard/df_test.csv")
path_definition_features_df = os.path.join(current_directory, "data/dashboard/definition_features.csv")
path_bureau = os.path.join(current_directory, "data/dashboard/bureau.csv")

df_train_full = pd.read_csv(path_df_train)
df_test = pd.read_csv(path_df_test)
definition_features_df = pd.read_csv(path_definition_features_df)
df_bureau = pd.read_csv(path_bureau)

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE_PATH = os.path.join(LOG_DIR, "dashboard.log")
logger = logging.getLogger("dashboard")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE_PATH)
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

st.set_page_config(layout="wide")

# ----- General styling for a more professional appearance -----
st.markdown(
    """
    <style>
    /* Overall background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #f7f7f7, #ffffff);
    }

    /* Tabs styling */
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-size: 14px;
        padding: 8px 16px;
        color: #444;
        background-color: #e0e0e0;
        border-radius: 4px 4px 0 0;
        margin-right: 2px;
        border: none;
        outline: none;
        cursor: pointer;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        background-color: #d4d4d4;
        color: #000;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #fff !important;
        border-bottom: 2px solid #444;
        color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----- Helper functions -----
def compute_color(probability):
    if 0 <= probability < 48:
        return "green"
    elif 48 <= probability <= 100:
        return "red"

def get_state():
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "data_received": False,
            "data": None,
            "last_sk_id_curr": None,
            "modified_data": {},
            "new_proba": None,
            # We'll store user's global font-size preference & chart color palette here
            "font_size": 14,  # default
            "palette_choice": "Standard",
        }
    elif "last_sk_id_curr" not in st.session_state["state"]:
        st.session_state["state"]["last_sk_id_curr"] = None
    return st.session_state["state"]

def radial_kpi(metric_label, metric_value, max_value=100):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=metric_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': metric_label},
            gauge={
                'axis': {'range': [0, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_value * 0.5], 'color': 'lightgreen'},
                    {'range': [max_value * 0.5, max_value], 'color': 'lightcoral'},
                ],
            }
        )
    )
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def simulate_probability_calculation(
    modified_features: dict,
    original_features: dict,
    shap_dict: dict,
    base_probability: float,
    df_train: pd.DataFrame,
) -> float:
    """
    Recalculates the probability of default given a set of modified feature values.
    This version applies a small multiplier to the SHAP-based offset to make
    changes slightly more pronounced.
    """
    total_offset = 0.0
    # This multiplier makes adjustments a bit larger without becoming extreme:
    sensitivity_multiplier = 7

    for feat, shap_val in shap_dict.items():
        if feat not in original_features or feat not in modified_features:
            continue
        if feat not in df_train.columns:
            continue

        old_val = original_features[feat]
        new_val = modified_features[feat]
        # Skip if any relevant value is None or NaN
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in [old_val, new_val, shap_val]):
            continue

        valid_series = df_train[feat].dropna()
        if valid_series.empty:
            continue

        feat_min = valid_series.min()
        feat_max = valid_series.max()
        range_val = max(1e-9, feat_max - feat_min)

        difference = new_val - old_val
        scaled_diff = difference / range_val

        # Apply a small multiplier so changes in value produce a slightly larger effect
        offset = shap_val * scaled_diff * sensitivity_multiplier
        total_offset += offset

    new_probability = base_probability + total_offset
    # Clamp the final probability between 0 and 100
    new_probability = max(0.0, min(100.0, new_probability))
    return new_probability


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("data/img/p7.png")

def feature_display_filter(feature_name, client_val):
    if pd.isna(client_val):
        return False
    if feature_name.startswith("EXT_SOURCE"):
        return False
    return True


# ----- Page header -----
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{img_base64}" style="width: 650px;">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center; color: black;'>Risk of Loan Default Estimation</h1>",
    unsafe_allow_html=True,
)

# ----- Session state & global user config -----
state = get_state()

# Global controls for accessibility
col_global_1, col_global_2 = st.columns([1,1])
with col_global_1:
    # Font size control
    font_size = st.slider(
        "Font Size",
        min_value=10,
        max_value=24,
        value=state["font_size"],
        help="Increase or decrease text size for better readability",
    )
    # Update in session state
    state["font_size"] = font_size

with col_global_2:
    palette_choice = st.radio(
        "Color Palette",
        options=["Standard", "Colorblind-Friendly"],
        index=0 if state["palette_choice"] == "Standard" else 1,
        help="Select colorblind-friendly visualization options",
    )
    state["palette_choice"] = palette_choice

# Modify the radial_kpi function:
def radial_kpi(metric_label, metric_value, max_value=100):
    state = get_state()
    if state["palette_choice"] == "Colorblind-Friendly":
        bar_color = "#4C78A8"  # Distinct blue
        steps = [
            {'range': [0, max_value*0.5], 'color': '#72B7B2'},  # Teal
            {'range': [max_value*0.5, max_value], 'color': '#E45756'},  # Red
        ]
    else:
        bar_color = "darkblue"
        steps = [
            {'range': [0, max_value*0.5], 'color': 'lightgreen'},
            {'range': [max_value*0.5, max_value], 'color': 'lightcoral'},
        ]
    
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=metric_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': metric_label},
            gauge={
                'axis': {'range': [0, max_value]},
                'bar': {'color': bar_color},
                'steps': steps,
            }
        )
    )
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Dynamically apply font size
st.markdown(
    f"""
    <style>
    html, body, [data-testid="stMarkdownContainer"] {{
        font-size: {state['font_size']}px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Dynamically apply tab size:
st.markdown(
    f"""
    <style>
    html, body, [data-testid="stMarkdownContainer"] {{
        font-size: {state['font_size']}px;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] {{
        font-size: {state['font_size']}px !important;
        padding: 8px 16px;
        color: #444;
        background-color: #e0e0e0;
        border-radius: 4px 4px 0 0;
        margin-right: 2px;
        border: none;
        outline: none;
        cursor: pointer;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
        background-color: #d4d4d4;
        color: #000;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: #fff !important;
        border-bottom: 2px solid #444;
        color: #000;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# For color palette toggles, define a small helper:
def get_palette():
    """Return a color scale based on user's palette choice."""
    if state["palette_choice"] == "Colorblind-Friendly":
        return px.colors.sequential.Greys  # or any colorblind-friendly scale
    else:
        return px.colors.sequential.Blues  # default

# ----- User input -----
sk_id_curr = st.text_input(
    "Enter SK_ID_CURR (e.g. 123456):",
    on_change=lambda: state.update(run=True)
)
col_run_button, col_spacer = st.columns([1, 20])

st.markdown(
    """
    <style>
        button {
            min-width: 200px !important;
            white-space: nowrap !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main logic for calling the API
if col_run_button.button("Run") or state["data_received"]:
    # Check if user typed a new ID
    if state["last_sk_id_curr"] != sk_id_curr:
        state["data_received"] = False
        state["last_sk_id_curr"] = sk_id_curr

    if not state["data_received"]:
        response = requests.post(
            "https://p7-api-e5hqc2b7ebbeb5gr.westeurope-01.azurewebsites.net/predict",
            json={"SK_ID_CURR": int(sk_id_curr)},
        )
        if response.status_code != 200:
            st.error(f"Error calling API: {response.status_code}")
            st.stop()

        state["data"] = response.json()
        state["data_received"] = True

    df_train = df_train_full.copy()
    data = state["data"]
    original_proba = data["probability"]
    feature_names = data["feature_names"]
    shap_values = data["shap_values"]
    feature_values = data["feature_values"]
    original_features = dict(zip(feature_names, feature_values))

    shap_values = [val[0] if isinstance(val, list) else val for val in shap_values]

    if not state["modified_data"]:
        for feat, val in zip(feature_names, feature_values):
            state["modified_data"][feat] = val

    if state["new_proba"] is None:
        state["new_proba"] = original_proba

    # ----- Decision and KPI -----
    color = compute_color(state["new_proba"])
    decision_message = "The loan will be granted." if state["new_proba"] < 48 else "The loan will not be granted."

    col_decision, col_kpi = st.columns([1, 1])
    with col_decision:
        st.markdown(
            f"<div style='text-align: center; color:{color}; font-size:24px;"
            f" border:2px solid {color}; padding:10px;'>{decision_message}</div>",
            unsafe_allow_html=True,
        )
    with col_kpi:
        st.plotly_chart(radial_kpi("Default Probability (%)", state["new_proba"], 100), use_container_width=True)

    # ----- Build filtered data series -----
    displayable_features = []
    for feat, val in zip(feature_names, feature_values):
        if feature_display_filter(feat, val):
            displayable_features.append((feat, val))
    data_series = pd.Series({f: v for f, v in displayable_features})

    # Create the tabs but we ONLY show Tab1 & Tab2 code
    st.write("---")
    tab1, tab3, tab4, tab5 = st.tabs(["Overview", "Analysis", "Credits", "Explicability"])

    # ----------------- Tab 1: Basic profile info -----------------
    with tab1:
        st.subheader("📌 Customer Profile & Loan Request")

        col_info_left, col_info_right = st.columns(2)

        # Basic calculations
        age_years = None
        if "DAYS_BIRTH" in data_series.index:
            age_years = -data_series["DAYS_BIRTH"] // 365

        employment_duration_years = None
        if "DAYS_EMPLOYED" in data_series.index:
            employment_duration_years = -data_series.get("DAYS_EMPLOYED", 0) // 365

        if "CODE_GENDER_M" in data_series.index:
            gender_pronoun = "👨 He" if data_series["CODE_GENDER_M"] else "👩 She"
        else:
            gender_pronoun = "❓ Unknown"

        def extract_category_value(ds, prefix):
            filtered = {key: val for key, val in ds.items() if key.startswith(prefix) and val == 1}
            if filtered:
                return list(filtered.keys())[0].replace(prefix + "_", "").lower()
            return "unknown"

        education_type = extract_category_value(data_series, "NAME_EDUCATION_TYPE")
        family_status = extract_category_value(data_series, "NAME_FAMILY_STATUS")
        housing_type = extract_category_value(data_series, "NAME_HOUSING_TYPE")
        income_type = extract_category_value(data_series, "NAME_INCOME_TYPE")

        def format_euros(val):
            return f"{val:,.2f} €" if pd.notna(val) else "N/A"

        amt_credit_val = data_series.get("AMT_CREDIT", np.nan)
        amt_annuity_val = data_series.get("AMT_ANNUITY", np.nan)
        amt_income_val = data_series.get("AMT_INCOME_TOTAL", np.nan)

        amt_credit = format_euros(amt_credit_val)
        amt_annuity = format_euros(amt_annuity_val)
        amt_income = format_euros(amt_income_val)

        # 🌟 **Profile Field Explanations**
        st.info(
            """
            📌 **Profile Field Explanations:**
            - 🆔 **Age**: The client's age, calculated from their date of birth.
            - 👤 **Gender**: The gender declared by the client.
            - 🏠 **Housing Status**: The type of housing the client resides in.
            - ❤️ **Marital Status**: The relationship status of the client (single, married, etc.).
            - 🎓 **Education Level**: The highest level of education completed by the client.
            - 💼 **Employment Type**: The type of job and source of income of the client.
            - 💰 **Annual Income**: The total income declared by the client.
            - 💳 **Loan Amount Requested**: The total amount of credit the client is applying for.
            - 📉 **Monthly Annuity**: The monthly repayment amount for the loan. A high annuity compared to income may indicate a higher risk of default.
            """
        )

        with col_info_left:
            st.markdown("### 👤 Client Info")
            if age_years is not None:
                st.markdown(f"**Gender:** {gender_pronoun}")
                st.markdown(f"**Age:** {int(age_years)} years old")
            if employment_duration_years is not None:
                st.markdown(f"**Employment:** {income_type}")
                st.markdown(f"- **Duration:** {employment_duration_years} years")

            st.markdown(f"🏠 **Housing:** {housing_type}")
            st.markdown(f"❤️ **Marital Status:** {family_status}")
            st.markdown(f"🎓 **Education:** {education_type}")

        with col_info_right:
            st.markdown("### 💳 Loan Details")
            st.markdown(
                f"""
                <div style="background-color:#f8f9fa; padding:10px; border-radius:10px;
                            border-left:5px solid #FF4B4B; margin-bottom:10px;">
                    <p style="font-size:16px; font-weight:bold;">💰 Loan Requested:
                        <span style="font-size:18px; color:#FF4B4B;">{amt_credit}</span>
                    </p>
                    <p style="font-size:16px;">📉 Monthly Annuity:
                        <span style="font-weight:bold;">{amt_annuity}</span>
                    </p>
                    <p style="font-size:16px;">💼 Annual Income:
                        <span style="font-weight:bold;">{amt_income}</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    # ----------------- Tab 3: Additional Visuals -----------------
    with tab3:
        # st.subheader("Additional Visuals")

        # Chart color & fullscreen toggles for this tab
        col_controls_t3a, col_controls_t3b = st.columns([1,1])
        with col_controls_t3a:
            palette_tab3 = st.radio(
                "Color Palette (Tab 3 Charts)",
                ("Standard", "Colorblind-Friendly"),
                help="Choose a palette for Tab 3 visuals (KPI, Sunburst, Bullet Charts)",
                key="palette_tab3"
            )

        # Helper to pick color scales based on palette_tab3
        def get_palette_tab3():
            if palette_tab3 == "Colorblind-Friendly":
                return px.colors.sequential.Greys  # example color scale
            else:
                return px.colors.sequential.Blues

        # # 1) Updated Probability KPI
        # col_kpi_new, col_dti = st.columns(2)
        # with col_kpi_new:
        #     st.markdown("##### Updated Probability (Currently)")
        #     # We'll keep the radial KPI as is, but you could adapt color scales if you wish
        #     st.plotly_chart(radial_kpi("Updated Default Probability (%)", state["new_proba"]), use_container_width=True)

        # # 2) Debt-to-Income KPI
        # with col_dti:
        #     st.markdown("##### Debt-to-Income Ratio")
        #     if "AMT_ANNUITY" in state["modified_data"] and "AMT_INCOME_TOTAL" in state["modified_data"]:
        #         annuity_val = state["modified_data"].get("AMT_ANNUITY", 0) or 0
        #         income_val = state["modified_data"].get("AMT_INCOME_TOTAL", 1e-9)
        #         dti_ratio = (annuity_val / income_val) * 100
        #         st.plotly_chart(radial_kpi("Debt-to-Income (%)", dti_ratio, 100), use_container_width=True)
        #     else:
        #         st.warning("No data for AMT_ANNUITY or AMT_INCOME_TOTAL.")

        # st.write("---")

    # ----------------- Univariate and Bivariate Analysis -----------------
        st.subheader("Univariate and Bivariate Analysis")

        # For easier reading, let's define toggles for color scale & fullscreen inside this tab
        col_controls_left, col_controls_right = st.columns([1,1])
        with col_controls_left:
            color_scale_uni_bi = st.radio(
                "Chart Palette (Uni/Bi)",
                options=["Default", "Colorblind-Friendly"],
                help="Choose a color palette for univariate/bivariate charts",
            )

        # We pick color palettes
        if color_scale_uni_bi == "Colorblind-Friendly":
            unibi_color_main = "#808080"  # some neutral grey
            unibi_color_highlight = "#505050"
        else:
            unibi_color_main = "#F9E3B1"  # the old default
            unibi_color_highlight = "#F8C471"

        # --- Layout columns for the two analyses
        col_uni, col_bi = st.columns(2)

        # -------------- Univariate --------------
        with col_uni:
            st.markdown("#### Univariate Analysis")
            numeric_series = data_series[data_series.apply(lambda x: isinstance(x, (int, float)))].to_frame().T
            if not numeric_series.empty:
                default_index_uni = 0
                if "AMT_INCOME_TOTAL" in numeric_series.columns:
                    default_index_uni = numeric_series.columns.get_loc("AMT_INCOME_TOTAL")

                valid_cols = [c for c in numeric_series.columns if not c.startswith("EXT_SOURCE")]
                if not valid_cols:
                    st.warning("No numeric features to display.")
                else:
                    selected_feature = st.selectbox(
                        "Select Feature:",
                        valid_cols,
                        index=min(default_index_uni, len(valid_cols)-1),
                        key="univariate_feature"
                    )
                    if selected_feature:
                        desc_match = definition_features_df[definition_features_df["Row"] == selected_feature]
                        if not desc_match.empty:
                            st.write(f"Feature: **{desc_match['Description'].iloc[0]}**")

                        col_data = df_train[selected_feature].dropna()
                        if col_data.empty:
                            st.warning("No data for this feature in df_train.")
                        else:
                            # Render chart in full or not
                            fig_size = (5,3)
                            fig_uni, ax_uni = plt.subplots(figsize=fig_size)
                            bins = np.linspace(col_data.min(), col_data.max(), 10)
                            counts_all, bins_all = np.histogram(col_data, bins=bins)
                            customer_value = data_series[selected_feature]
                            customer_bin_index = np.digitize(customer_value, bins=bins) - 1

                            ax_uni.bar(
                                bins_all[:-1],
                                counts_all,
                                width=(bins_all[1] - bins_all[0]),
                                color=unibi_color_main,
                                alpha=0.8,
                                label='All Clients'
                            )
                            if 0 <= customer_bin_index < len(counts_all):
                                ax_uni.bar(
                                    bins_all[customer_bin_index],
                                    counts_all[customer_bin_index],
                                    width=(bins_all[1] - bins_all[0]),
                                    color=unibi_color_highlight,
                                    label='Current Customer'
                                )
                            ax_uni.set_facecolor('#FCFCEF')
                            ax_uni.set_xlabel(selected_feature, fontsize=7)
                            ax_uni.set_ylabel('Count (Log Scale)', fontsize=7)
                            ax_uni.set_title(f'Distribution of {selected_feature}', fontsize=9)
                            ax_uni.set_yscale('log')
                            ax_uni.legend(fontsize=6, loc='upper right')
                            plt.tight_layout()
                            st.pyplot(fig_uni)

        # -------------- Bivariate --------------
        with col_bi:
            st.markdown("#### Bivariate Analysis")
            numeric_df_train = df_train.select_dtypes(include=[np.number])
            numeric_df_train = numeric_df_train[[c for c in numeric_df_train.columns if not c.startswith("EXT_SOURCE")]]
            if not numeric_df_train.empty:
                f1_index = 0
                f2_index = 0
                if "AMT_INCOME_TOTAL" in numeric_df_train.columns:
                    f1_index = numeric_df_train.columns.get_loc("AMT_INCOME_TOTAL")
                if "AMT_ANNUITY" in numeric_df_train.columns:
                    f2_index = numeric_df_train.columns.get_loc("AMT_ANNUITY")

                all_cols = numeric_df_train.columns.tolist()
                if not all_cols:
                    st.warning("No numeric features for bivariate.")
                else:
                    feature1 = st.selectbox("Select Feature 1:", all_cols, index=min(f1_index, len(all_cols)-1), key="bivariate_feature1")
                    feature2 = st.selectbox("Select Feature 2:", all_cols, index=min(f2_index, len(all_cols)-1), key="bivariate_feature2")

                    desc_f1 = definition_features_df[definition_features_df["Row"] == feature1]
                    desc_f2 = definition_features_df[definition_features_df["Row"] == feature2]
                    if not desc_f1.empty:
                        st.write(f"Feature 1: **{desc_f1['Description'].iloc[0]}**")
                    if not desc_f2.empty:
                        st.write(f"Feature 2: **{desc_f2['Description'].iloc[0]}**")

                    def generate_bivariate_plot(dfref, feat1, feat2, sk_id):
                        plt.clf()
                        if feat1 not in dfref.columns or feat2 not in dfref.columns:
                            st.error("Selected features not in df_train.")
                            return
                        dfref = dfref[[feat1, feat2, "SK_ID_CURR"]].dropna()
                        if dfref.empty:
                            st.warning("No data after dropping NaNs.")
                            return
                        client_data = dfref[dfref['SK_ID_CURR'] == int(sk_id)]
                        if client_data.empty:
                            st.error(f"No data for SK_ID_CURR {sk_id}.")
                            return

                        client_f1 = client_data[feat1].iloc[0]
                        client_f2 = client_data[feat2].iloc[0]

                        fig_size_bi = (5,3)
                        fig_bi, ax_bi = plt.subplots(figsize=fig_size_bi)
                        ax_bi.scatter(
                            dfref[feat1], dfref[feat2],
                            color=unibi_color_main,
                            alpha=0.6,
                            label="All Clients",
                            s=10
                        )
                        ax_bi.scatter(
                            client_f1, client_f2,
                            color=unibi_color_highlight,
                            marker='o',
                            s=50,
                            label='Current Client'
                        )
                        ax_bi.set_xscale('log')
                        ax_bi.set_yscale('log')
                        ax_bi.set_xlabel(feat1, fontsize=7)
                        ax_bi.set_ylabel(feat2, fontsize=7)
                        ax_bi.set_title(f"{feat1} vs. {feat2}", fontsize=9, fontweight='bold')
                        ax_bi.legend(fontsize=6, loc='upper right')
                        plt.tight_layout()
                        st.pyplot(fig_bi)

                    generate_bivariate_plot(numeric_df_train, feature1, feature2, sk_id_curr)


        st.write("---")

        # 3) Sunburst: Income & Expenses
        col_sunburst, col_bullet = st.columns([1.5, 1])
        with col_sunburst:
            st.markdown("#### Sunburst: Income & Expenses")
            # We also add a fullscreen toggle & color palette for the sunburst
            # but let's keep it simple. We'll do an example snippet:
            def display_sunburst_for_client(ds):
                data_sunburst = {'Category': [], 'SubCategory': [], 'Value': []}
                income_total = ds.get("AMT_INCOME_TOTAL", 0)
                data_sunburst['Category'].append("Income")
                data_sunburst['SubCategory'].append("AMT_INCOME_TOTAL")
                data_sunburst['Value'].append(income_total)

                annuity_val = ds.get("AMT_ANNUITY", 0)
                data_sunburst['Category'].append("Expenses")
                data_sunburst['SubCategory'].append("Loan_Annuity")
                data_sunburst['Value'].append(annuity_val)

                child_count = ds.get("CNT_CHILDREN", 0)
                child_expense_est = child_count * 200
                data_sunburst['Category'].append("Expenses")
                data_sunburst['SubCategory'].append("Child_Expenses (est.)")
                data_sunburst['Value'].append(child_expense_est)

                df_sun = pd.DataFrame(data_sunburst)

                fig_sun = px.sunburst(
                    df_sun,
                    path=['Category', 'SubCategory'],
                    values='Value',
                    color='Category',
                    title="Approx. Income & Expenses Breakdown",
                    color_discrete_sequence=get_palette_tab3()
                )
                st.plotly_chart(fig_sun)

            display_sunburst_for_client(state["modified_data"])

        # 4) Bullet Charts
        with col_bullet:
            st.markdown("#### Bullet Charts")
            st.info("""
            **Bullet Chart Explanations**:
            - This chart compares the client's actual value (dark bar) against a median or target (red marker).
            - Use it to see if the client's metric is above or below typical thresholds.
            """)

            def bullet_chart(actual, target, name="Metric"):
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[target * 1.2],
                    y=[name],
                    marker_color='lightgray',
                    orientation='h',
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[target, target],
                    y=[-0.2, 0.2],
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False
                ))
                fig.add_trace(go.Bar(
                    x=[actual],
                    y=[name],
                    orientation='h',
                    marker_color='darkblue',
                    showlegend=False
                ))
                fig.update_layout(
                    barmode='overlay',
                    height= 100,
                    margin=dict(l=60, r=60, t=30, b=30),
                    xaxis=dict(showgrid=False, zeroline=False),
                )
                return fig

            if "AMT_CREDIT" in state["modified_data"]:
                bc_client_credit = state["modified_data"]["AMT_CREDIT"] or 0
                bc_target_credit = df_train["AMT_CREDIT"].dropna().median() if "AMT_CREDIT" in df_train.columns else 0
                fig_bc_credit = bullet_chart(bc_client_credit, bc_target_credit, name="AMT_CREDIT (Median)")
                st.plotly_chart(fig_bc_credit, use_container_width=True)

            if "AMT_INCOME_TOTAL" in state["modified_data"]:
                bc_client_inc = state["modified_data"]["AMT_INCOME_TOTAL"] or 0
                bc_target_inc = df_train["AMT_INCOME_TOTAL"].dropna().median() if "AMT_INCOME_TOTAL" in df_train.columns else 0
                fig_bc_income = bullet_chart(bc_client_inc, bc_target_inc, name="AMT_INCOME_TOTAL (Median)")
                st.plotly_chart(fig_bc_income, use_container_width=True)

        st.write("---")


    # ================== Tab 4: Credits (continuation) ==================
    with tab4:
        st.subheader("Open Credits & Credit History")

        # Local toggles for palette & fullscreen in this tab
        col_tab4_1, col_tab4_2 = st.columns([1,1])
        with col_tab4_1:
            palette_credits = st.radio(
                "Color Palette (Credits)",
                ("Standard", "Colorblind-Friendly"),
                help="Choose a color palette for the credit charts",
                key="palette_credits"
            )

        # Helper to pick color scale for credits tab
        def get_palette_credits():
            if palette_credits == "Colorblind-Friendly":
                return px.colors.sequential.Greys  # or any colorblind-friendly scale
            else:
                return px.colors.sequential.Blues

        # Explanations for columns, aimed at the bank agent
        st.info("""
        **Credit Columns Explanation**:
        - **CREDIT_ACTIVE**: Whether the credit is Active, Closed, etc.
        - **AMT_CREDIT_SUM_DEBT**: Total debt remaining on this line of credit.
        - **AMT_CREDIT_SUM_LIMIT**: The credit limit (if applicable).
        - **CREDIT_TYPE**: Type of credit product (e.g. credit card, car loan, etc.).
        - **DAYS_CREDIT_ENDDATE**: Days until the scheduled end date of the credit.
        """)

        # 1) Check if df_bureau is available
        if "df_bureau" not in globals() and "df_bureau" not in locals():
            st.warning("Credit data is not available (no 'df_bureau' found).")
        else:
            df_bureau_user = df_bureau[df_bureau["SK_ID_CURR"] == int(sk_id_curr)].copy()
            if df_bureau_user.empty:
                st.info("No bureau data found for this user.")
            elif "CREDIT_ACTIVE" not in df_bureau_user.columns:
                st.info("Cannot determine active credits (column 'CREDIT_ACTIVE' is missing).")
            else:
                df_active_credits = df_bureau_user[df_bureau_user["CREDIT_ACTIVE"] == "Active"]
                if df_active_credits.empty:
                    st.info("No open (active) credits found for this user.")
                else:
                    # Card-based layout
                    st.markdown('<div class="credits-container">', unsafe_allow_html=True)

                    has_debt_col = "AMT_CREDIT_SUM_DEBT" in df_active_credits.columns
                    has_sum_col  = "AMT_CREDIT_SUM" in df_active_credits.columns
                    has_limit_col= "AMT_CREDIT_SUM_LIMIT" in df_active_credits.columns

                    total_open_credits = len(df_active_credits)
                    total_debt = df_active_credits["AMT_CREDIT_SUM_DEBT"].fillna(0).sum() if has_debt_col else 0
                    total_credit_sum = df_active_credits["AMT_CREDIT_SUM"].fillna(0).sum() if has_sum_col else 0
                    total_limit = df_active_credits["AMT_CREDIT_SUM_LIMIT"].fillna(0).sum() if has_limit_col else 0

                    def format_euros(val):
                        return f"{val:,.0f} €" if pd.notna(val) else "N/A"

                    # Card 1: Basic open-credit summary
                    st.markdown(
                        f"""
                        <div class="credit-card">
                            <div class="credit-card-title">Open Credits Summary</div>
                            <p><strong>Number of Active Credits:</strong> {total_open_credits}</p>
                            <p><strong>Total Debt:</strong> <span class="highlight">{format_euros(total_debt) if has_debt_col else "N/A"}</span></p>
                            <p><strong>Total Credit Amount:</strong> {format_euros(total_credit_sum) if has_sum_col else "N/A"}</p>
                            <p><strong>Credit Limit (All Lines):</strong> {format_euros(total_limit) if has_limit_col else "N/A"}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Card 2: Overdue details
                    has_overdue_col     = "AMT_CREDIT_SUM_OVERDUE" in df_active_credits.columns
                    has_max_overdue_col = "AMT_CREDIT_MAX_OVERDUE" in df_active_credits.columns
                    has_day_overdue_col = "CREDIT_DAY_OVERDUE" in df_active_credits.columns

                    overdue_sum = df_active_credits["AMT_CREDIT_SUM_OVERDUE"].fillna(0).sum() if has_overdue_col else 0
                    max_overdue = df_active_credits["AMT_CREDIT_MAX_OVERDUE"].fillna(0).max() if has_max_overdue_col else 0
                    day_overdue_any = False
                    if has_day_overdue_col:
                        day_overdue_any = (df_active_credits["CREDIT_DAY_OVERDUE"] > 0).any()

                    st.markdown(
                        f"""
                        <div class="credit-card">
                            <div class="credit-card-title">Overdue Details</div>
                            <p><strong>Total Overdue Amount:</strong> <span class="highlight">{format_euros(overdue_sum) if has_overdue_col else "N/A"}</span></p>
                            <p><strong>Max Overdue Observed:</strong> {format_euros(max_overdue) if has_max_overdue_col else "N/A"}</p>
                            <p><strong>Currently Overdue Credits:</strong> {"Yes" if day_overdue_any else ("No" if has_day_overdue_col else "N/A")}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("</div>", unsafe_allow_html=True)  # closes .credits-container

                    # Detailed table
                    st.markdown("### Detailed View of Active Credits")
                    columns_to_show = [
                        "CREDIT_TYPE", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM",
                        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT",
                        "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG",
                        "DAYS_CREDIT_ENDDATE",
                    ]
                    columns_existing = [col for col in columns_to_show if col in df_active_credits.columns]
                    if columns_existing:
                        df_to_show = df_active_credits[columns_existing]
                        numeric_cols = df_to_show.select_dtypes(include=[np.number]).columns
                        # Possibly enlarge in fullscreen
                        st.dataframe(df_to_show.style.format("{:,.2f}", subset=numeric_cols))
                    else:
                        st.info("No relevant credit columns found (they may be missing).")

                    # Visuals: Debt by credit type
                    st.write("---")
                    st.subheader("Debt Distribution by Credit Type")
                    if ("CREDIT_TYPE" in df_active_credits.columns) and ("AMT_CREDIT_SUM_DEBT" in df_active_credits.columns):
                        debt_by_type = (
                            df_active_credits.groupby("CREDIT_TYPE")["AMT_CREDIT_SUM_DEBT"]
                            .sum()
                            .reset_index(name="TotalDebt")
                        )
                        if not debt_by_type.empty:
                            fig_debt_type = px.bar(
                                debt_by_type,
                                x="CREDIT_TYPE",
                                y="TotalDebt",
                                title="Total Debt by Credit Type",
                                labels={"CREDIT_TYPE": "Credit Type", "TotalDebt": "Total Debt (€)"},
                                color="TotalDebt",
                                color_continuous_scale=get_palette_credits()
                            )
                            st.plotly_chart(fig_debt_type)
                        else:
                            st.info("No debt data available by credit type.")
                    else:
                        st.info("Required columns ('CREDIT_TYPE' or 'AMT_CREDIT_SUM_DEBT') missing; cannot plot debt distribution.")


    ###########################
    # Tab 5: SHAP sliders (Features)
    ###########################
    with tab5:
        st.header("What-If Analysis: Key Features")
        st.write("Below are features strongly impacting risk. You can adjust them to see how your probability might change (hypothetical).")

        # Local toggles for palette & fullscreen in SHAP
        col_tab5_1, col_tab5_2 = st.columns([1,1])
        with col_tab5_1:
            palette_shap = st.radio(
                "Color Palette (SHAP Features)",
                ("Standard", "Colorblind-Friendly"),
                help="Choose a color palette for SHAP bar charts if needed",
                key="palette_shap"
            )

        # 1) Define fallback descriptions for features with 'No description available'
        fallback_descriptions = {
            "PAYMENT_RATE": "Payment rate used to compare the installment amount with the total credit amount.",
            "INSTAL_AMT_PAYMENT_SUM": "Total sum of installment payments made for a given loan or period.",
            "CODE_GENDER_F": "Indicator for female gender (1 = female).",
            "ANNUITY_INCOME_PERC": "Ratio of the client’s annuity to their total income.",
            "BURO_DAYS_CREDIT_MAX": "Maximum number of days related to any credit found in the bureau data.",
            "INSTAL_PAYMENT_DIFF_MEAN": "Average difference between expected and actual installment payments.",
            "BURO_AMT_CREDIT_SUM_MAX": "Maximum credit amount found among all bureau records.",
            "WALLSMATERIAL_MODE_Panel": "Indicates if the dwelling’s wall material is of type Panel.",
            "INSTAL_DPD_MEAN": "Mean days past due across the client's installment payments.",
            "NAME_FAMILY_STATUS_Married": "Indicator that the client’s family status is Married.",
            "ACTIVE_DAYS_CREDIT_MAX": "Maximum duration (in days) of any active credit in the bureau records."
        }

        # 2) Prepare SHAP data
        shap_data = []
        for feat, shap_val, fval in zip(feature_names, shap_values, feature_values):
            if feature_display_filter(feat, fval):
                shap_data.append((feat, shap_val))

        if not shap_data:
            st.warning("No valid SHAP data to show.")
        else:
            # Convert our shap_data into a DataFrame
            shap_df_filtered = pd.DataFrame(shap_data, columns=["Feature","SHAP Value"]).sort_values(
                by="SHAP Value", ascending=False
            )

            # Identify top features that increase risk vs. decrease risk
            top_positive_shap = shap_df_filtered.head(10).copy()
            top_negative_shap = shap_df_filtered.tail(10).copy()

            # 3) Helper function to fetch the Description from definition_features_df,
            #    then use fallback if needed
            def get_feature_description(feature_name):
                match_row = definition_features_df[definition_features_df["Row"] == feature_name]
                if not match_row.empty:
                    desc = match_row["Description"].iloc[0]
                    # If the original file says 'No description available', use our fallback
                    if desc.strip().lower() == "no description available":
                        return fallback_descriptions.get(feature_name, "No description available (no fallback text).")
                    else:
                        return desc
                else:
                    # If the feature is not in definition_features_df, also try fallback
                    return fallback_descriptions.get(feature_name, "No description available (no fallback text).")

            # Add a new 'Description' column to each subset
            top_positive_shap["Description"] = top_positive_shap["Feature"].apply(get_feature_description)
            top_negative_shap["Description"] = top_negative_shap["Feature"].apply(get_feature_description)

            # 4) Prepare a dictionary of SHAP values for recalculation
            shap_dict = dict(zip(shap_df_filtered["Feature"], shap_df_filtered["SHAP Value"]))

            # Retrieve client data from df_train
            input_data = df_train[df_train['SK_ID_CURR'] == int(sk_id_curr)].copy()
            if input_data.empty:
                st.error("No matching data for this client in df_train.")
            else:
                col_pos, col_neg = st.columns(2)

                ########################
                # FEATURES INCREASING RISK
                ########################
                with col_pos:
                    st.subheader("Features Increasing Risk")
                    # Display Feature and Description side by side
                    for feature in top_positive_shap["Feature"]:
                        if feature not in df_train.columns:
                            continue
                        desc = get_feature_description(feature)
                        st.write(f"**{feature}** — {desc}")

                        current_val = state["modified_data"].get(feature, np.nan)
                        unique_vals = df_train[feature].dropna().unique()

                        if len(set(unique_vals)) == 2 and set(unique_vals).issubset({0,1}):
                            # Binary feature
                            default_index = 1 if current_val == 1 else 0
                            chosen_radio = st.radio(
                                label="",
                                options=["No", "Yes"],
                                index=default_index,
                                key=f"{feature}_pos_radio",
                                horizontal=True
                            )
                            new_val = 1 if chosen_radio == "Yes" else 0
                            state["modified_data"][feature] = new_val

                        else:
                            # Numeric or continuous feature
                            col_data = df_train[feature].dropna()
                            if col_data.empty:
                                new_val = st.number_input(
                                    label="",
                                    value=float(current_val),
                                    key=f"{feature}_pos_num"
                                )
                                state["modified_data"][feature] = new_val
                            else:
                                global_min = col_data.min()
                                global_max = col_data.max()
                                new_val = st.slider(
                                    label="",
                                    min_value=float(global_min),
                                    max_value=float(global_max),
                                    value=float(current_val),
                                    label_visibility="hidden",
                                    key=f"{feature}_pos_slider"
                                )
                                state["modified_data"][feature] = new_val

                ########################
                # FEATURES DECREASING RISK
                ########################
                with col_neg:
                    st.subheader("Features Decreasing Risk")
                    for feature in top_negative_shap["Feature"]:
                        if feature not in df_train.columns:
                            continue
                        desc = get_feature_description(feature)
                        st.write(f"**{feature}** — {desc}")

                        current_val = state["modified_data"].get(feature, np.nan)
                        unique_vals = df_train[feature].dropna().unique()

                        if len(set(unique_vals)) == 2 and set(unique_vals).issubset({0,1}):
                            # Binary feature
                            default_index = 1 if current_val == 1 else 0
                            chosen_radio = st.radio(
                                label="",
                                options=["No","Yes"],
                                index=default_index,
                                key=f"{feature}_neg_radio",
                                horizontal=True
                            )
                            new_val = 1 if chosen_radio == "Yes" else 0
                            state["modified_data"][feature] = new_val

                        else:
                            # Numeric or continuous feature
                            col_data = df_train[feature].dropna()
                            if col_data.empty:
                                new_val = st.number_input(
                                    label="",
                                    value=float(current_val),
                                    key=f"{feature}_neg_num"
                                )
                                state["modified_data"][feature] = new_val
                            else:
                                global_min = col_data.min()
                                global_max = col_data.max()
                                new_val = st.slider(
                                    label="",
                                    min_value=float(global_min),
                                    max_value=float(global_max),
                                    value=float(current_val),
                                    label_visibility="hidden",
                                    key=f"{feature}_neg_slider"
                                )
                                state["modified_data"][feature] = new_val

                # 5) Recalculation button
                st.write("---")
                if st.button("Recalculate Probability"):
                    updated_shap_dict = dict(zip(shap_df_filtered["Feature"], shap_df_filtered["SHAP Value"]))
                    state["new_proba"] = simulate_probability_calculation(
                        modified_features=state["modified_data"],
                        original_features=original_features,
                        shap_dict=updated_shap_dict,
                        base_probability=original_proba,
                        df_train=df_train
                    )

                    # Display updated probability with a new radial KPI
                    st.success(f"New Probability: {state['new_proba']:.2f}%")
                    st.plotly_chart(radial_kpi("Updated Default Probability (%)", state["new_proba"], 100), use_container_width=True)
                    st.stop()
