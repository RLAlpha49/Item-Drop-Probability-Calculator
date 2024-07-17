import streamlit as st
from scipy.stats import binom
import pandas as pd

# Title of the app
st.title("Item Drop Probability Calculator")

# Input fields for user
p = st.number_input("Enter the probability of success (e.g., 0.2 for 20%)", min_value=0.0, max_value=1.0, step=0.01)
desired_successes = st.number_input("Enter the desired number of successes", min_value=1, step=1)
played_attempts = st.number_input(
    "Enter the number of times you have played the mission (optional)", min_value=0, step=1
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


# Calculate probabilities and display table
if p and desired_successes:
    # Probability thresholds to display
    thresholds = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]

    # Table to store results
    result_table = []

    for threshold in thresholds:
        for n in range(1, 1000):  # Arbitrary upper limit for search
            if 1 - binom.cdf(desired_successes - 1, n, p) >= threshold:
                if total_mission_time > 0:
                    total_time = n * total_mission_time
                    formatted_time = format_time(total_time)
                    result_table.append([int(threshold * 100), n, formatted_time])
                else:
                    result_table.append([int(threshold * 100), n])
                break

    # Convert result table to DataFrame
    if total_mission_time > 0:
        df = pd.DataFrame(
            result_table, columns=["Probability (%)", "Number of Attempts", "Total Time (minutes:seconds)"]
        )
    else:
        df = pd.DataFrame(result_table, columns=["Probability (%)", "Number of Attempts"])

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
            width: 80%; /* Adjust the width as needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display result table
    st.write("### Number of Mission Attempts Required for Different Success Probabilities")
    # Convert DataFrame to HTML without index
    html = df.to_html(index=False)

    # Display the HTML table in Streamlit without the index column
    st.markdown(html, unsafe_allow_html=True)

    # If played attempts are provided, calculate the probability of achieving the desired successes
    if played_attempts > 0:
        prob_successes = (1 - binom.cdf(desired_successes - 1, played_attempts, p)) * 100
        st.write(
            f"The probability of getting at least {desired_successes} items in {played_attempts} attempts is {prob_successes:.4f}%"
        )
