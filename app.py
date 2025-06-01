import math
from io import BytesIO
import numpy as np
import streamlit as st
from scipy.stats import binom
import pandas as pd
import altair as alt


def format_time(minutes):
    """Convert minutes to a string format of hours:minutes:seconds if >= 1 hour, else minutes:seconds."""
    total_seconds = int(minutes * 60)
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


@st.cache_data
def calculate_probabilities(prob, successes, mission_time, max_attempts_input, display_all_thresholds=False):
    """
    Calculates probabilities for mission success based on given parameters.
    Args:
        prob (float): The probability of success for a single attempt.
        successes (int): The number of successes required.
        mission_time (float): The time required for each mission attempt.
        max_attempts_input (int): The maximum number of attempts to consider.
        display_all_thresholds (bool): Whether to show all thresholds (1% increments) in the display table.
    Returns:
        tuple: A tuple containing:
            - list: A table of probabilities for display thresholds.
            - list: A detailed table of probabilities for CSV export.
            - list: A detailed table of probabilities for each number of attempts.
            - int: The maximum number of attempts, rounded up.
            - numpy.ndarray: An array of probabilities for each number of attempts.
    """
    if display_all_thresholds:
        display_thresholds = [i / 100 for i in range(1, 100)]  # 1% increments
    else:
        display_thresholds = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    csv_thresholds = [i / 100 for i in range(1, 100)]
    attempts = np.arange(1, max_attempts_input + 1)
    cdf_values = binom.cdf(successes - 1, attempts, prob)
    probabilities = 1 - cdf_values

    table = []
    table_csv = []
    max_attempt_found = 0
    # For each threshold, always add a row, even if not reached
    for threshold in csv_thresholds:
        idx = np.searchsorted(probabilities, threshold, side="left")
        if idx < len(attempts):
            # Interpolate for decimal attempts if not exactly at threshold
            if idx == 0 or probabilities[idx] == threshold:
                n_decimal = attempts[idx]
            else:
                # Linear interpolation between idx-1 and idx
                p1, p2 = probabilities[idx - 1], probabilities[idx]
                a1, a2 = attempts[idx - 1], attempts[idx]
                n_decimal = a1 + (threshold - p1) * (a2 - a1) / (p2 - p1)
            max_attempt_found = max(max_attempt_found, n_decimal)
            if mission_time > 0:
                total_time = n_decimal * mission_time
                formatted_time = format_time(total_time)
                table_csv.append([int(threshold * 100), n_decimal, formatted_time])
                if threshold in display_thresholds:
                    table.append([int(threshold * 100), n_decimal, formatted_time])
            else:
                table_csv.append([int(threshold * 100), n_decimal])
                if threshold in display_thresholds:
                    table.append([int(threshold * 100), n_decimal])
        else:
            # Not reached within max_attempts_input
            if mission_time > 0:
                table_csv.append([int(threshold * 100), "Too many", "Too long"])
                if threshold in display_thresholds:
                    table.append([int(threshold * 100), "Too many", "Too long"])
            else:
                table_csv.append([int(threshold * 100), "Too many"])
                if threshold in display_thresholds:
                    table.append([int(threshold * 100), "Too many"])
    # If no threshold was reached, set max_attempts to the max checked
    if max_attempt_found == 0:
        max_attempts = len(attempts)
    else:
        max_attempts = max_attempt_found

    # Add detailed probabilities for each number of attempts to the CSV table
    detailed_table_csv = []
    for n in range(1, int(np.ceil(max_attempts)) + 1):
        prob_val = probabilities[n - 1] * 100
        if mission_time > 0:
            total_time = n * mission_time
            formatted_time = format_time(total_time)
            detailed_table_csv.append([n, prob_val, formatted_time])
        else:
            detailed_table_csv.append([n, prob_val])

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


# Title of the app
st.title("Item Drop Probability Calculator")

# --- Multiple Item Types Support ---
if "item_list" not in st.session_state:
    st.session_state.item_list = [
        {
            "name": "Item 1",
            "p": 20.0,
            "desired_successes": 1,
            "played_attempts": 0,
            "mission_minutes": 5,
            "mission_seconds": 0,
        }
    ]

# Add Item button with a more visually distinct style
st.markdown('<div style="text-align:right; margin-bottom: 0.5em;">', unsafe_allow_html=True)
add_item_clicked = st.button("➕ Add Item", key="add_item_btn")
st.markdown("</div>", unsafe_allow_html=True)
if add_item_clicked:
    st.session_state.item_list.append(
        {
            "name": f"Item {len(st.session_state.item_list) + 1}",
            "p": 20.0,
            "desired_successes": 1,
            "played_attempts": 0,
            "mission_minutes": 5,
            "mission_seconds": 0,
        }
    )

# Ensure item_result_selector is initialized before using in expander
if "item_result_selector" not in st.session_state:
    st.session_state.item_result_selector = 0

items_to_delete = []
for idx, item in enumerate(st.session_state.item_list):
    row_cols = st.columns([1, 12])
    with row_cols[0]:
        if st.button("🗑️", key=f"delete_{idx}", help="Delete this item", disabled=len(st.session_state.item_list) == 1):
            items_to_delete.append(idx)
    with row_cols[1]:
        with st.expander(f"{item['name']} Settings", expanded=(idx == st.session_state.item_result_selector)):
            cols = st.columns([6, 1])
            with cols[0]:
                item["name"] = st.text_input("Item Name", value=item["name"], key=f"name_{idx}")
                item["p"] = st.number_input(
                    f"Probability of success for {item['name']} (e.g., 20 for 20%)",
                    min_value=0.0,
                    step=0.01,
                    format="%.3f",
                    value=item["p"],
                    key=f"p_{idx}",
                    help="The chance of getting the item in a single attempt, as a percentage.",
                )
                item["desired_successes"] = st.number_input(
                    f"Desired number of successes for {item['name']}",
                    min_value=1,
                    step=1,
                    value=item["desired_successes"],
                    key=f"ds_{idx}",
                    help="How many times you want to obtain the item.",
                )
                item["played_attempts"] = st.number_input(
                    f"Number of attempts already made for {item['name']} (optional)",
                    min_value=0,
                    step=1,
                    value=item["played_attempts"],
                    key=f"pa_{idx}",
                    help="If you have already tried some attempts, enter that number here.",
                )
                item["mission_minutes"] = st.number_input(
                    f"Mission time - minutes for {item['name']} (optional)",
                    min_value=0,
                    step=1,
                    value=item["mission_minutes"],
                    key=f"mm_{idx}",
                    help="Minutes per mission attempt.",
                )
                item["mission_seconds"] = st.number_input(
                    f"Mission time - seconds for {item['name']} (optional)",
                    min_value=0,
                    max_value=59,
                    step=1,
                    value=item["mission_seconds"],
                    key=f"ms_{idx}",
                    help="Seconds per mission attempt.",
                )
# Actually delete after the loop to avoid index issues
deleted_selected = False
for idx in sorted(items_to_delete, reverse=True):
    if idx == st.session_state.item_result_selector:
        deleted_selected = True
    del st.session_state.item_list[idx]
# Ensure the selected index is valid after deletion and rerun if needed
if "item_result_selector" in st.session_state:
    if st.session_state.item_result_selector >= len(st.session_state.item_list):
        st.session_state.item_result_selector = max(0, len(st.session_state.item_list) - 1)
        st.rerun()
    elif deleted_selected:
        st.session_state.item_result_selector = min(
            st.session_state.item_result_selector, len(st.session_state.item_list) - 1
        )
        st.rerun()

# Shared max attempts input
max_attempts_input = st.number_input(
    "Maximum number of attempts to consider",
    min_value=1,
    max_value=100_000_000,
    value=1_000_000,
    step=1_000,
    help="Increase this if you want to see probabilities for more attempts. Very large values may take a long time to compute.",
)
if max_attempts_input > 10_000_000:
    st.warning("Calculating probabilities for more than 10 million attempts may take a long time!")

# Input validation and warnings (after all inputs)
for idx, item in enumerate(st.session_state.item_list):
    if item["p"] == 0:
        st.warning(f"Probability of success for {item['name']} cannot be zero.")
    if (
        item["desired_successes"] > 0
        and item["played_attempts"] > 0
        and item["desired_successes"] > item["played_attempts"]
    ):
        st.warning(f"Desired successes for {item['name']} cannot exceed the number of attempts.")
    if item["p"] > 100:
        st.warning(f"Probability for {item['name']} cannot exceed 100%.")


# Results section for each item
def render_item_results(item, max_attempts_input):
    """
    Render results and statistics for a given item based on mission parameters.

    Args:
        item (dict): Item details including:
            - name (str): Item name
            - mission_minutes (int): Mission duration in minutes
            - mission_seconds (int): Additional mission seconds
            - p (float): Success probability percentage
            - desired_successes (int): Target number of successes
            - played_attempts (int): Attempts already played
        max_attempts_input (int): Maximum attempts to consider

    Returns:
        None: Updates the Streamlit UI with results, charts, and downloadable files.
    """
    total_mission_time = item["mission_minutes"] + item["mission_seconds"] / 60
    p = item["p"] / 100 if item["p"] > 0 else 0
    desired_successes = item["desired_successes"]
    played_attempts = item["played_attempts"]
    if p and desired_successes:
        st.success(f"Calculation complete for {item['name']}! See results below.")
        st.write(f"### Number of Mission Attempts Required for Different Success Probabilities for {item['name']}")
        show_inbetween = st.checkbox(
            f"Show all probabilities (1% increments) for {item['name']}",
            value=False,
            key=f"inbetween_{item['name']}",
            help="If checked, the table will show every 1% probability threshold instead of just the main ones.",
        )
        display_table, csv_table, detailed_table_csv, max_attempts, probabilities = calculate_probabilities(
            p, desired_successes, total_mission_time, max_attempts_input, show_inbetween
        )
        # Warn if even after max_attempts_input attempts, probability is extremely low
        show_table_and_graph = True
        if max_attempts >= max_attempts_input and (not display_table or not csv_table):
            st.warning(
                f"Even after the maximum number of attempts, the probability of reaching your goal for {item['name']} is extremely low. Try lowering the number of desired successes, increasing the probability, or increasing the maximum number of attempts to continue the calculation."
            )
            show_table_and_graph = False
        if show_table_and_graph:
            if total_mission_time > 0:
                df_display = pd.DataFrame(
                    display_table,
                    columns=["Probability (%)", "Number of Attempts", "Total Time (hours:minutes:seconds)"],
                )
                df_csv = pd.DataFrame(
                    csv_table, columns=["Probability (%)", "Number of Attempts", "Total Time (hours:minutes:seconds)"]
                )
                df_detailed_csv = pd.DataFrame(
                    detailed_table_csv,
                    columns=["Number of Attempts", "Probability (%)", "Total Time (hours:minutes:seconds)"],
                )
            else:
                df_display = pd.DataFrame(display_table, columns=["Probability (%)", "Number of Attempts"])
                df_csv = pd.DataFrame(csv_table, columns=["Probability (%)", "Number of Attempts"])
                df_detailed_csv = pd.DataFrame(detailed_table_csv, columns=["Number of Attempts", "Probability (%)"])
            # Checkbox for showing decimals in the Number of Attempts column
            show_decimals = st.checkbox(
                "Show decimals for Number of Attempts", value=False, key=f"show_decimals_{item['name']}"
            )
            df_display_to_show = df_display.copy()
            if not show_decimals and "Number of Attempts" in df_display_to_show.columns:
                # Only round numeric values, leave 'Too many' as is
                df_display_to_show["Number of Attempts"] = df_display_to_show["Number of Attempts"].apply(
                    lambda x: int(round(x)) if isinstance(x, (int, float)) and not isinstance(x, bool) else x
                )
            st.dataframe(df_display_to_show, use_container_width=True, height=422)
            # If played attempts are provided, calculate the probability of achieving the desired successes
            if played_attempts > 0:
                from scipy.stats import binom

                prob_successes = (1 - binom.cdf(desired_successes - 1, played_attempts, p)) * 100
                st.write(
                    f"The probability of getting at least {desired_successes} {item['name']} in {played_attempts} attempts is {prob_successes:.4f}%"
                )
            # Cumulative probability area chart using Altair
            with st.spinner(f"Generating cumulative probability chart for {item['name']}..."):
                import numpy as np

                chart_df = pd.DataFrame(
                    {
                        "Attempts": np.arange(1, max_attempts + 1),
                        "Cumulative Probability (%)": probabilities[:max_attempts] * 100,
                    }
                )
                st.write(f"#### Cumulative Probability of Achieving {desired_successes} {item['name']}(s)")
                chart_type_area = st.checkbox(
                    "Show as area chart", value=False, key=f"area_chart_toggle_{item['name']}"
                )
                if chart_type_area:
                    chart = (
                        alt.Chart(chart_df)
                        .mark_area(
                            line={"color": "#1f77b4"},
                            color=alt.Gradient(
                                gradient="linear",
                                stops=[
                                    alt.GradientStop(color="#1f77b4", offset=0),
                                    alt.GradientStop(color="white", offset=1),
                                ],
                                x1=1,
                                x2=1,
                                y1=1,
                                y2=0,
                            ),
                            interpolate="monotone",
                        )
                        .encode(
                            x=alt.X("Attempts", title="Number of Attempts"),
                            y=alt.Y(
                                "Cumulative Probability (%)",
                                title="Cumulative Probability (%)",
                                scale=alt.Scale(domain=[0, 100]),
                            ),
                            tooltip=["Attempts", "Cumulative Probability (%)"],
                        )
                        .properties(height=500)
                    )
                else:
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line(color="#1f77b4", interpolate="monotone")
                        .encode(
                            x=alt.X("Attempts", title="Number of Attempts"),
                            y=alt.Y(
                                "Cumulative Probability (%)",
                                title="Cumulative Probability (%)",
                                scale=alt.Scale(domain=[0, 100]),
                            ),
                            tooltip=["Attempts", "Cumulative Probability (%)"],
                        )
                        .properties(height=500)
                    )
                st.altair_chart(chart, use_container_width=True)
            # Advanced Statistical Features: Expected Value, Variance, Confidence Interval, Monte Carlo Simulation
            st.write("---")
            st.write(f"### Advanced Statistics for {item['name']}")
            # Use the greatest number in the 'Number of Attempts' column in the table
            if "Number of Attempts" in df_display.columns:
                numeric_attempts = pd.to_numeric(df_display["Number of Attempts"], errors="coerce")
                n = int(numeric_attempts.dropna().max()) if not numeric_attempts.dropna().empty else max_attempts_input
            else:
                n = max_attempts_input
            p_single = item["p"] / 100 if item["p"] > 0 else 0
            expected_value = n * p_single
            variance = n * p_single * (1 - p_single)
            stddev = variance**0.5
            ci_low = expected_value - 1.96 * stddev
            ci_high = expected_value + 1.96 * stddev
            st.markdown(
                rf"""
                **Expected number of successes:**
                $$
                E[X] = n \times p = {n} \times {p_single:.4f} = {expected_value:.2f}
                $$
                """
            )
            st.markdown(
                rf"""
                **Variance:**
                $$
                \mathrm{{Var}}[X] = n \times p \times (1-p) = {n} \times {p_single:.4f} \times (1 - {p_single:.4f}) = {variance:.2f}
                $$
                """
            )
            st.markdown(
                rf"""
                **95% Confidence Interval for mean:**
                $$
                [{expected_value:.2f} - 1.96 \times {stddev:.2f},\ {expected_value:.2f} + 1.96 \times {stddev:.2f}] = [{ci_low:.2f},\ {ci_high:.2f}]
                $$
                """
            )

            # Monte Carlo Simulation
            st.write("---")
            st.write(f"#### Monte Carlo Simulation for {item['name']}")
            sim_attempts = st.number_input(
                f"Number of attempts per simulation for {item['name']} (leave blank or 0 to use table value)",
                min_value=0,
                max_value=100_000_000,
                value=0,
                step=1,
                key=f"sim_attempts_{item['name']}",
            )
            n_sim = int(sim_attempts) if sim_attempts else n
            sim_runs = st.number_input(
                f"Number of simulation runs for {item['name']}",
                min_value=1000,
                max_value=1_000_000,
                value=10000,
                step=1000,
                key=f"sim_runs_{item['name']}",
            )
            run_sim = st.button(f"Run Simulation for {item['name']}", key=f"run_sim_{item['name']}")
            if run_sim:
                import numpy as np

                successes = 0
                for _ in range(int(sim_runs)):
                    attempts = np.random.binomial(n_sim, p_single)
                    if attempts >= item["desired_successes"]:
                        successes += 1
                empirical_prob = successes / sim_runs * 100
                # Theoretical probability for comparison
                from scipy.stats import binom

                theoretical_prob = (1 - binom.cdf(item["desired_successes"] - 1, n_sim, p_single)) * 100
                st.markdown(
                    rf"""
                    **Empirical probability:**
                    $$
                    \frac{{\text{{Number of successful simulations}}}}{{\text{{Total simulations}}}} \times 100 = \frac{{{successes}}}{{{sim_runs}}} \times 100 = {empirical_prob:.4f}\%
                    $$
                    """
                )
                st.markdown(
                    rf"""
                    **Theoretical probability:**
                    $$
                    1 - F(k-1; n, p) = 1 - \text{{binom.cdf}}({item["desired_successes"] - 1},\ {n_sim},\ {p_single:.4f}) = {theoretical_prob:.4f}\%
                    $$
                    """
                )
            # Prepare advanced statistics and simulation results for export
            stats_rows = [
                ["Expected number of successes", f"E[X] = n * p = {n} * {p_single:.4f} = {expected_value:.2f}"],
                ["Variance", f"Var[X] = n * p * (1-p) = {n} * {p_single:.4f} * (1 - {p_single:.4f}) = {variance:.2f}"],
                [
                    "95% Confidence Interval for mean",
                    f"[{expected_value:.2f} - 1.96 * {stddev:.2f}, {expected_value:.2f} + 1.96 * {stddev:.2f}] = [{ci_low:.2f}, {ci_high:.2f}]",
                ],
            ]
            # If simulation was run, add those results
            if "sim_results" not in st.session_state:
                st.session_state.sim_results = {}
            sim_key = f"sim_{item['name']}"
            sim_result = st.session_state.sim_results.get(sim_key, None)
            if run_sim:
                st.session_state.sim_results[sim_key] = {
                    "successes": successes,
                    "sim_runs": sim_runs,
                    "empirical_prob": empirical_prob,
                    "theoretical_prob": theoretical_prob,
                    "n_sim": n_sim,
                    "desired_successes": item["desired_successes"],
                    "p_single": p_single,
                }
                sim_result = st.session_state.sim_results[sim_key]
            # Always display the most recent simulation results if available
            if sim_result:
                st.markdown(
                    rf"""
                    **Empirical probability:**
                    $$
                    \frac{{\text{{Number of successful simulations}}}}{{\text{{Total simulations}}}} \times 100 = \frac{{{sim_result["successes"]}}}{{{sim_result["sim_runs"]}}} \times 100 = {sim_result["empirical_prob"]:.4f}\%
                    $$
                    """
                )
                st.markdown(
                    rf"""
                    **Theoretical probability:**
                    $$
                    1 - F(k-1; n, p) = 1 - \text{{binom.cdf}}({sim_result["desired_successes"] - 1},\ {sim_result["n_sim"]},\ {sim_result["p_single"]:.4f}) = {sim_result["theoretical_prob"]:.4f}\%
                    $$
                    """
                )
                stats_rows.append(
                    [
                        "Empirical probability",
                        f"({sim_result['successes']} / {sim_result['sim_runs']}) * 100 = {sim_result['empirical_prob']:.4f}% for at least {sim_result['desired_successes']} successes in {sim_result['n_sim']} attempts",
                    ]
                )
                stats_rows.append(
                    [
                        "Theoretical probability",
                        f"1 - binom.cdf({sim_result['desired_successes'] - 1}, {sim_result['n_sim']}, {sim_result['p_single']:.4f}) = {sim_result['theoretical_prob']:.4f}%",
                    ]
                )
            stats_df = pd.DataFrame(stats_rows, columns=["Statistic", "Calculation"])

            # Generate the Excel file
            excel_file = None
            excel_error = None
            try:
                # Add stats_df as a new sheet
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df_detailed_csv.to_excel(writer, index=False, sheet_name="Detailed Probabilities")
                    df_csv.to_excel(writer, index=False, sheet_name="Display Table")
                    stats_df.to_excel(writer, index=False, sheet_name="Advanced Statistics")
                output.seek(0)
                excel_file = output.getvalue()
            except ValueError as e:
                if "sheet is too large" in str(e):
                    excel_error = (
                        f"The result table for {item['name']} is too large to export to Excel. "
                        "Excel sheets have a maximum of 1,048,576 rows. "
                        "Try reducing the maximum number of attempts or exporting a smaller result."
                    )
                else:
                    excel_error = f"Could not generate Excel file for {item['name']}: {e}"
            if excel_error:
                st.warning(excel_error)
            elif excel_file is not None:
                st.download_button(
                    label=f"Download results for {item['name']} as Excel",
                    data=excel_file,
                    file_name=f"results_{item['name'].replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                # Add ZIP download button with three CSVs (display, detailed, stats)
                import io
                import zipfile

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    # Display Table CSV
                    display_csv = df_display_to_show.to_csv(index=False)
                    zf.writestr("display_table.csv", display_csv)
                    # Detailed Probabilities CSV
                    detailed_csv = df_detailed_csv.to_csv(index=False)
                    zf.writestr("detailed_probabilities.csv", detailed_csv)
                    # Advanced Statistics CSV
                    stats_csv = stats_df.to_csv(index=False)
                    zf.writestr("advanced_statistics.csv", stats_csv)
                zip_buffer.seek(0)
                st.download_button(
                    label=f"Download results for {item['name']} as ZIP of CSVs",
                    data=zip_buffer,
                    file_name=f"results_{item['name'].replace(' ', '_')}.zip",
                    mime="application/zip",
                )


# Selector for which item's results to show
item_names = [item["name"] for item in st.session_state.item_list]

# Use wider columns for navigation buttons
col_prev, col_select, col_next = st.columns([0.15, 0.7, 0.15])
with col_select:
    selected_idx = st.selectbox(
        "Select item to view results",
        options=list(range(len(item_names))),
        format_func=lambda i: item_names[i],
        key="item_result_selector",
        index=st.session_state.item_result_selector,
    )
with col_prev:
    if st.button("Previous", disabled=st.session_state.item_result_selector == 0):
        st.session_state.item_result_selector = max(0, st.session_state.item_result_selector - 1)
        st.rerun()
with col_next:
    if st.button("Next", disabled=st.session_state.item_result_selector == len(item_names) - 1):
        st.session_state.item_result_selector = min(len(item_names) - 1, st.session_state.item_result_selector + 1)
        st.rerun()

item = st.session_state.item_list[st.session_state.item_result_selector]
render_item_results(item, max_attempts_input)

# Add general CSS for all Streamlit buttons to prevent wrapping and set min width
st.markdown(
    """
    <style>
    button.stButton {
        min-width: 100px !important;
        white-space: nowrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
