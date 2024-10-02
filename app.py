import numpy as np
import streamlit as st
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
import math
from io import BytesIO

# Title of the app
st.title("Item Drop Probability Calculator")

# Input fields for user
p = st.number_input("Enter the probability of success (e.g., 20 for 20%)", min_value=0.0, step=0.01, format="%.3f")
desired_successes = st.number_input("Enter the desired number of successes", min_value=1, step=1)
played_attempts = st.number_input(
    "Enter the number of times you have already attempted (optional)", min_value=0, step=1
)

# Input for mission time
mission_minutes = st.number_input("Enter the mission time - minutes (optional)", min_value=0, step=1)
mission_seconds = st.number_input("Enter the mission time - seconds (optional)", min_value=0, max_value=59, step=1)

# Convert mission time to total minutes
total_mission_time = mission_minutes + mission_seconds / 60


def format_time(minutes):
    """Convert minutes to a string format of minutes:seconds."""
    total_seconds = int(minutes * 60)
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{mins}:{secs:02d}"


@st.cache_data
def calculate_probabilities(prob, successes, mission_time):
    # Original specific probabilities to display
    """
    Calculates probabilities for mission success based on given parameters.

    Args:
        prob (float): The probability of success for a single attempt.
        successes (int): The number of successes required.
        mission_time (float): The time required for each mission attempt.

    Returns:
        tuple: A tuple containing:
            - list: A table of probabilities for display thresholds.
            - list: A detailed table of probabilities for CSV export.
            - list: A detailed table of probabilities for each number of attempts.
            - int: The maximum number of attempts, rounded up.
            - numpy.ndarray: An array of probabilities for each number of attempts.
    """
    display_thresholds = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    # Detailed probabilities for CSV
    csv_thresholds = [i / 100 for i in range(1, 100)]

    # Precompute binomial CDF for all possible attempts
    max_attempts = 0
    attempts = np.arange(1, 1000000 + 1)
    cdf_values = binom.cdf(successes - 1, attempts, prob)
    probabilities = 1 - cdf_values

    # Use search sorted to find the minimum number of attempts for each threshold
    table = []
    table_csv = []
    for threshold in csv_thresholds:
        idx = np.searchsorted(probabilities, threshold, side="left")
        if idx < len(attempts):
            n = attempts[idx]
            if mission_time > 0:
                total_time = n * mission_time
                formatted_time = format_time(total_time)
                table_csv.append([int(threshold * 100), n, formatted_time])
                if threshold in display_thresholds:
                    table.append([int(threshold * 100), n, formatted_time])
            else:
                table_csv.append([int(threshold * 100), n])
                if threshold in display_thresholds:
                    table.append([int(threshold * 100), n])
                    max_attempts = n

    # Add detailed probabilities for each number of attempts to the CSV table
    detailed_table_csv = []
    for n in range(1, max_attempts + 1):
        prob = probabilities[n - 1] * 100
        if mission_time > 0:
            total_time = n * mission_time
            formatted_time = format_time(total_time)
            detailed_table_csv.append([n, prob, formatted_time])
        else:
            detailed_table_csv.append([n, prob])

    # Round up max_attempts
    if max_attempts < 100:
        max_attempts = math.ceil(max_attempts / 10) * 10
    elif max_attempts < 1000:
        max_attempts = math.ceil(max_attempts / 100) * 100
    elif max_attempts < 10000:
        max_attempts = math.ceil(max_attempts / 1000) * 1000
    elif max_attempts < 100000:
        max_attempts = math.ceil(max_attempts / 10000) * 10000
    elif max_attempts < 1000000:
        max_attempts = math.ceil(max_attempts / 100000) * 100000
    else:
        max_attempts = math.ceil(max_attempts / 1000000) * 1000000

    return table, table_csv, detailed_table_csv, max_attempts, probabilities


@st.cache_data
def generate_excel_file(df_detailed_csv, df_csv):
    """Generates an Excel file containing two sheets from provided DataFrames.

    Args:
        df_detailed_csv (pandas.DataFrame): DataFrame containing detailed probabilities data.
        df_csv (pandas.DataFrame): DataFrame containing display table data.

    Returns:
        BytesIO: A bytes buffer containing the generated Excel file with two sheets.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_detailed_csv.to_excel(writer, index=False, sheet_name="Detailed Probabilities")
        df_csv.to_excel(writer, index=False, sheet_name="Display Table")
    output.seek(0)
    return output


# Convert probability to a percentage if it's greater than 0
if p > 0:
    p = p / 100

# Calculate probabilities and display table
if p and desired_successes:
    display_table, csv_table, detailed_table_csv, max_attempts, probabilities = calculate_probabilities(
        p, desired_successes, total_mission_time
    )

    # Convert display table to DataFrame
    if total_mission_time > 0:
        df_display = pd.DataFrame(
            display_table, columns=["Probability (%)", "Number of Attempts", "Total Time (minutes:seconds)"]
        )
    else:
        df_display = pd.DataFrame(display_table, columns=["Probability (%)", "Number of Attempts"])

    # Convert CSV table to DataFrame
    if total_mission_time > 0:
        df_csv = pd.DataFrame(
            csv_table, columns=["Number of Attempts", "Probability (%)", "Total Time (minutes:seconds)"]
        )
        df_detailed_csv = pd.DataFrame(
            detailed_table_csv, columns=["Number of Attempts", "Probability (%)", "Total Time (minutes:seconds)"]
        )
    else:
        df_csv = pd.DataFrame(csv_table, columns=["Number of Attempts", "Probability (%)"])
        df_detailed_csv = pd.DataFrame(detailed_table_csv, columns=["Number of Attempts", "Probability (%)"])

    st.markdown(
        """
        <style>
        h3 {
            text-align: center;
        }
        th, td {
            text-align: right !important;
        }
        table {
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display result table
    with st.spinner("Generating display table..."):
        st.write("### Number of Mission Attempts Required for Different Success Probabilities")
        # Convert DataFrame to HTML without index
        html = df_display.to_html(index=False)

        # Display the HTML table in Streamlit without the index column
        st.markdown(html, unsafe_allow_html=True)

    # If played attempts are provided, calculate the probability of achieving the desired successes
    if played_attempts > 0:
        prob_successes = (1 - binom.cdf(desired_successes - 1, played_attempts, p)) * 100
        st.write(
            f"The probability of getting at least {desired_successes} items in {played_attempts} attempts is {prob_successes:.4f}%"
        )

    # Plot the probability distribution
    with st.spinner("Generating graph..."):
        fig, ax = plt.subplots()
        x = np.arange(1, max_attempts + 1)
        y = probabilities[:max_attempts]
        ax.plot(x, y, label="Probability Distribution")
        ax.set_xlabel("Number of Attempts")
        ax.set_ylabel("Probability")
        ax.set_title("Probability Distribution of Successes")
        ax.legend(loc="upper left")

        st.pyplot(fig)

    # Generate the Excel file
    excel_file = generate_excel_file(df_detailed_csv, df_csv)

    st.download_button(
        label="Download results as Excel",
        data=excel_file,
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
