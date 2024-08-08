import streamlit as st
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
import math

# Title of the app
st.title("Item Drop Probability Calculator")

# Input fields for user
p = st.number_input("Enter the probability of success (e.g., 20 for 20%)", min_value=0.0, step=0.01, format="%.2f")
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
    display_thresholds = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    # Detailed probabilities for CSV
    csv_thresholds = [i / 100 for i in range(1, 100)]

    # Table to store results for display
    table = []
    # Table to store results for CSV
    table_csv = []
    attempts_max = 0

    for threshold in csv_thresholds:
        for n in range(1, 100000):  # Arbitrary upper limit for search
            if 1 - binom.cdf(successes - 1, n, prob) >= threshold:
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
                if threshold == 0.99:
                    attempts_max = n
                break

    # Round up max_attempts
    if attempts_max < 100:
        attempts_max = math.ceil(attempts_max / 10) * 10
    elif attempts_max < 1000:
        attempts_max = math.ceil(attempts_max / 100) * 100
    elif attempts_max < 10000:
        attempts_max = math.ceil(attempts_max / 1000) * 1000
    else:
        attempts_max = math.ceil(attempts_max / 10000) * 10000

    return table, table_csv, attempts_max


# Convert probability to a percentage if it's greater than 0
if p > 0:
    p = p / 100

# Calculate probabilities and display table
if p and desired_successes:
    display_table, csv_table, max_attempts = calculate_probabilities(p, desired_successes, total_mission_time)

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
            csv_table, columns=["Probability (%)", "Number of Attempts", "Total Time (minutes:seconds)"]
        )
    else:
        df_csv = pd.DataFrame(csv_table, columns=["Probability (%)", "Number of Attempts"])

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
    fig, ax = plt.subplots()
    x = range(1, max_attempts)
    y = [1 - binom.cdf(desired_successes - 1, n, p) for n in x]
    ax.plot(x, y, label="Probability Distribution")
    ax.set_xlabel("Number of Attempts")
    ax.set_ylabel("Probability")
    ax.set_title("Probability Distribution of Successes")
    ax.legend()

    st.pyplot(fig)

    # Add a download button for the CSV file
    csv = df_csv.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="results.csv",
        mime="text/csv",
    )
