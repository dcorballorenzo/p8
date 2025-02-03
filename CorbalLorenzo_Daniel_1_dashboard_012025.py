import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import logging

# Absolute path of the current script file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Path to the training dataset file 'df_train.csv' located in the 'data' subdirectory
# and feature definitions file 'definition_features.csv'
path_df_train = os.path.join(current_directory, "data/dashboard/df_train.csv")
path_definition_features_df = os.path.join(current_directory, "data/dashboard/definition_features.csv")

# Loading data in specified paths
df_train = pd.read_csv(path_df_train)
definition_features_df = pd.read_csv(path_definition_features_df)

#################### LOGS #################################
# Create a log file if it does not exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Definition of the file where the logging will appear
LOG_FILE_PATH = os.path.join(LOG_DIR, "dashboard.log")

# Create a logger and set its level (DEBUG)
logger = logging.getLogger("dashboard")
logger.setLevel(logging.DEBUG)

# File handler to write log messages to a file
file_handler = logging.FileHandler(LOG_FILE_PATH)
# Prevent duplicate handlers
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler only if it doesn't exist
    logger.addHandler(file_handler)

# Set the page layout to wide mode
st.set_page_config(layout="wide")

##############################################################################################
def get_title_font_size(height):
    base_size = 12  # Base font size
    scale_factor = height / 600.0  # Scale factor based on default height
    return base_size * scale_factor


def generate_figure(df, title_text, x_anchor, yaxis_categoryorder, yaxis_side):
    fig = go.Figure(data=[go.Bar(y=df["Feature"], x=df["SHAP Value"], orientation="h")])
    annotations = generate_annotations(df, x_anchor)

    title_font_size = get_title_font_size(600)
    fig.update_layout(
        annotations=annotations,
        title_text=title_text,
        title_x=0.25,
        title_y=0.88,
        title_font=dict(size=title_font_size),
        yaxis=dict(
            categoryorder=yaxis_categoryorder, side=yaxis_side, tickfont=dict(size=14)
        ),
        height=600,
    )
    fig.update_xaxes(title_text="Impact des fonctionnalités")
    return fig


def generate_annotations(df, x_anchor):
    annotations = []
    for y_val, x_val, feat_val in zip(
        df["Feature"], df["SHAP Value"], df["Feature Value"]
    ):
        formatted_feat_val = (
            feat_val
            if pd.isna(feat_val)
            else (int(feat_val) if feat_val == int(feat_val) else feat_val)
        )
        annotations.append(
            dict(
                x=x_val,
                y=y_val,
                text=f"<b>{formatted_feat_val}</b>",
                showarrow=False,
                xanchor=x_anchor,
                yanchor="middle",
                font=dict(color="white"),
            )
        )
    return annotations


def compute_color(value):
    if 0 <= value < 48:
        return "green"
    elif 48 <= value <= 100:
        return "red"


def format_value(val):
    if pd.isna(val):
        return val
    if isinstance(val, (float, int)):
        if val == int(val):
            return int(val)
        return round(val, 2)
    return val


def find_closest_description(feature_name, definitions_df):
    for index, row in definitions_df.iterrows():
        if row["Row"] in feature_name:
            return row["Description"]
    return None


def plot_distribution(selected_feature, col):
    if selected_feature:
        data = df_train[selected_feature]

        # Find the feature value for the current client
        client_feature_value = feature_values[feature_names.index(selected_feature)]

        fig = go.Figure()

        # Check if the feature is categorical
        unique_values = sorted(data.dropna().unique())
        if set(unique_values) <= {0, 1, 2, 3, 4, 5, 6, 7}:
            # Count occurrences of each value
            counts = data.value_counts().sort_index()

            # Ensure lengths match
            assert len(unique_values) == len(counts)

            # Modify the color list to match the size of unique_values
            colors = ["blue"] * len(unique_values)

            # Update client_value
            client_value = (
                unique_values.index(client_feature_value)
                if client_feature_value in unique_values
                else None
            )

            # Update the corresponding color if client_value is not None
            if client_value is not None:
                colors[client_value] = "red"

            # Plot using unique_values
            fig.add_trace(go.Bar(x=unique_values, y=counts.values, marker_color=colors))

        else:
            # Calculate bins for the histogram
            hist_data, bins = np.histogram(data.dropna(), bins=20)

            # Find the bin for client_feature_value
            client_bin_index = np.digitize(client_feature_value, bins) - 1

            # Create a list of colors for the bins
            colors = ["blue"] * len(hist_data)
            if (
                0 <= client_bin_index < len(hist_data)
            ):  # Ensure the index is valid
                colors[client_bin_index] = "red"

            # Plot the distribution for continuous variables
            fig.add_trace(
                go.Histogram(
                    x=data,
                    marker=dict(color=colors, opacity=0.7),
                    name="Distribution",
                    xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
                )
            )

            # Use a logarithmic scale if the distribution is highly skewed
            mean_val = np.mean(hist_data)
            std_val = np.std(hist_data)
            if std_val > 3 * mean_val:  # Adjust this threshold as needed
                fig.update_layout(yaxis_type="log")

        height = 600  # Adjust this value based on the default height of your figure
        title_font_size = get_title_font_size(height)

        fig.update_layout(
            title_text=f"Distribution pour {selected_feature}",
            title_font=dict(size=title_font_size),
            xaxis_title=selected_feature,
            yaxis_title="Nombre de clients",
            title_x=0.3,
        )

        col.plotly_chart(fig, use_container_width=True)

        # Display the definition of the selected feature
        description = find_closest_description(selected_feature, definition_features_df)
        if description:
            col.write(f"**Definition:** {description}")


# Function to retrieve stored states
def get_state():
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "data_received": False,
            "data": None,
            "last_sk_id_curr": None,  # Store the last submitted ID
        }
    elif (
        "last_sk_id_curr" not in st.session_state["state"]
    ):  # Check if 'last_sk_id_curr' exists
        st.session_state["state"][
            "last_sk_id_curr"
        ] = None  # If not, add it.

    return st.session_state["state"]


def create_gauge_chart(probability):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Probabilité de non-remboursement (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "black"},  # Arrow color
            "steps": [
                {"range": [0, 48], "color": "green"},
                {"range": [48, 100], "color": "red"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.8,
                "value": probability
            }
        }
    ))

    return fig


state = get_state()

st.markdown(
    "<h1 style='text-align: center; color: black;'>Estimation du risque de non-remboursement</h1>",
    unsafe_allow_html=True,
)
sk_id_curr = st.text_input(
    "Entrez le SK_ID_CURR (ex: 123456):", on_change=lambda: state.update(run=True)
)
col1, col2 = st.columns([1, 20])

st.markdown(
    """
    <style>
        /* Style pour le bouton */
        button {
            width: 60px !important;
            white-space: nowrap !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if col1.button("Run") or state["data_received"]:
    # Before making the API call, check if the current ID is different from the last ID
    if state["last_sk_id_curr"] != sk_id_curr:
        state["data_received"] = False
        state["last_sk_id_curr"] = sk_id_curr  # Update the last ID

    if not state["data_received"]:
        # Update the API endpoint to the remote URL
        response = requests.post(
            "https://p7-api-e5hqc2b7ebbeb5gr.westeurope-01.azurewebsites.net/predict",
            json={"SK_ID_CURR": int(sk_id_curr)},
        )
        if response.status_code != 200:
            st.error(f"Erreur lors de l'appel à l'API: {response.status_code}")
            st.stop()

        state["data"] = response.json()
        state["data_received"] = True

    data = state["data"]
    # logger.debug("DATA: %s", data)

    proba = data["probability"]
    feature_names = data["feature_names"]
    shap_values = data["shap_values"]
    # logger.debug("SHAP values: %s", shap_values)
    feature_values = data["feature_values"]
    shap_values = [val[0] if isinstance(val, list) else val for val in shap_values]
    shap_df = pd.DataFrame(
        list(
            zip(
                feature_names,
                shap_values,
                [format_value(val) for val in feature_values],
            )
        ),
        columns=["Feature", "SHAP Value", "Feature Value"],
    )


    # =========================================================================
    # CREDIT DECISION & RISK VISUALIZATION
    # =========================================================================
    color = compute_color(proba)
    st.empty()

    decision_message = (
        "Le prêt sera accordé." if proba < 48 else "Le prêt ne sera pas accordé."
    )
    st.markdown(
        f"<div style='text-align: center; color:{color}; font-size:30px; border:2px solid {color}; padding:10px;'>{decision_message}</div>",
        unsafe_allow_html=True,
    )

    col2.plotly_chart(create_gauge_chart(proba), use_container_width=True)


    # =========================================================================
    # CUSTOMERS BASIC INFORMATIONS
    # =========================================================================
    # Display customer information
    data_series = pd.Series(data=data['feature_values'], index=data['feature_names'])

    st.header("Customer Information:")

    age_years = -data_series['DAYS_BIRTH'] // 365  # Calculate age from DAYS_BIRTH
    employment_duration_years = -data_series.get('DAYS_EMPLOYED', 0) // 365  # Calculate employment duration from DAYS_EMPLOYED

    ### Gender
    gender_pronoun = "He" if data_series['CODE_GENDER_M'] else "She"
    gender2_pronoun = "his" if data_series['CODE_GENDER_M'] else "her"

    def extract_category_value(data_series, prefix):
        """
        Extracts the category value from data_series based on a given prefix.
        Returns the cleaned category name in lowercase or "Unknown" if no valid entry is found.
        """
        filtered = {key: val for key, val in data_series.items() if key.startswith(prefix) and val == 1}

        if filtered:
            return list(filtered.keys())[0].replace(prefix + "_", "").lower()
        return "unknown"

    education_type = extract_category_value(data_series, "NAME_EDUCATION_TYPE")
    family_status = extract_category_value(data_series, "NAME_FAMILY_STATUS")
    housing_type = extract_category_value(data_series, "NAME_HOUSING_TYPE")
    income_type = extract_category_value(data_series, "NAME_INCOME_TYPE")
    contract_type = extract_category_value(data_series, "NAME_CONTRACT_TYPE")

    customer_description = f"""
    **{gender_pronoun.capitalize()}** is a **{int(age_years)}** years old **{education_type}** who works in the **{income_type}** sector. **{gender_pronoun.capitalize()}** lives in a **{housing_type}** and is currently **{employment_duration_years}** years into employment. **{gender_pronoun.capitalize()}** is **{family_status}** and has applied for a **{contract_type.lower()}** loan. **{gender2_pronoun.capitalize()}** income is **{data_series['AMT_INCOME_TOTAL']}** €.
    """
    st.write(customer_description)

    loan_description = f"""
    The loan asked is **{data_series['AMT_CREDIT']}** €, and the annuity asked are **{data_series['AMT_ANNUITY']}** €.
    """
    st.write(loan_description)

    ###################################### WIP #########################################
    # =========================================================================
    # COMPARATIVE ANALYSIS USING GRAPHS
    # ========================================================================
    # st.header('III. Comparative Analysis')
    # st.subheader('III.1. Univariate Analysis')
    # # Get all features (assuming numerical features)
    # all_features = data_series.select_dtypes(include=[np.number]) # Adjust for categorical features if needed
    # # Filter controls
    # selected_feature = st.selectbox('Select Feature:', all_features.columns, index=all_features.columns.get_loc('AMT_INCOME_TOTAL'))  # Set default


    # # Load feature descriptions (assuming customer_data_description is a Pandas DataFrame)
    # feature_descriptions = customer_data_description

    # # Select feature
    # all_features = list(feature_descriptions["Row"])  # Assuming "Row" contains feature names

    # # Find description for the selected feature
    # feature_description = feature_descriptions[feature_descriptions["Row"] == selected_feature]["Description"].iloc[0]

    # # Print description
    # st.write(f"Feature : **{feature_description}**")



    # # Filter data based on selected feature
    # filtered_data = data_series.copy() # Avoid modifying original data

    # # Separate data for full dataset and current customer
    # full_data_values = np.array(customer_data_copy[selected_feature])
    # customer_value = customer_data_copy[selected_feature].iloc[customer_index]


    # # Create bins (adjust number of bins as needed)

    # bins = np.linspace(filtered_data[selected_feature].min(), filtered_data[selected_feature].max(), 10)  # 10 bins
    # # Calculate bin width (assuming equally spaced bins)
    # bin_width = bins[1] - bins[0]

    # # Count data points within each bin for all customers and the selected customer
    # counts_all, bins_all = np.histogram(filtered_data[selected_feature], bins=bins)
    # count_customer, _ = np.histogram(filtered_data[selected_feature].iloc[customer_index], bins=bins)


    # # Find the bin index for the customer value
    # customer_bin_index = np.digitize(customer_value, bins=bins) - 1  # Adjust for zero-based indexing

    # # Create bar chart with bins and log scale on y-axis
    # fig, ax = plt.subplots()
    # ax.bar(bins_all[:-1] + bin_width/2, counts_all, width=bin_width, color='gray', alpha=0.7, label='All Clients')
    # ax.bar(bins_all[customer_bin_index] + bin_width/2, counts_all, width=bin_width, color='red', label='Current Customer')  # Use customer_bin_index
    # ax.set_xlabel(selected_feature)  # Adjust label based on feature
    # ax.set_ylabel('Count (Log Scale)')  # Update label
    # ax.set_title(f'Distribution of {selected_feature} (Binned)')
    # ax.set_yscale('log')  # Set log scale for y-axis
    # ax.legend()
    # plt.tight_layout()
    # st.pyplot(plt.gcf())

    # # Define top_positive_shap and top_negative_shap
    # top_positive_shap = shap_df.sort_values(by="SHAP Value", ascending=False).head(10)
    # top_negative_shap = shap_df.sort_values(by="SHAP Value").head(10)

    fig_positive = generate_figure(
        top_positive_shap,
        "Top 10 des fonctionnalités augmentant le risque de non-remboursement",
        "right",
        "total ascending",
        "left",
    )
    fig_negative = generate_figure(
        top_negative_shap,
        "Top 10 des fonctionnalités réduisant le risque de non-remboursement",
        "left",
        "total descending",
        "right",
    )

    # Create a new row for the charts
    col_chart1, col_chart2 = st.columns(2)
    col_chart1.plotly_chart(fig_positive, use_container_width=True)
    col_chart2.plotly_chart(fig_negative, use_container_width=True)

    # Create columns for the dropdowns
    col1, col2 = st.columns(2)

    # Place the first dropdown in col1
    with col1:
        selected_feature_positive = st.selectbox(
            "Sélectionnez une fonctionnalité augmentant le risque",
            [""] + top_positive_shap["Feature"].tolist(),
        )

    # Place the second dropdown in col2
    with col2:
        selected_feature_negative = st.selectbox(
            "Sélectionnez une fonctionnalité réduisant le risque",
            [""] + top_negative_shap["Feature"].tolist(),
        )

    # Call the `plot_distribution` functions
    plot_distribution(selected_feature_positive, col1)
    plot_distribution(selected_feature_negative, col2)
