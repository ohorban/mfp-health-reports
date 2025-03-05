import os
import subprocess
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from jinja2 import Environment, FileSystemLoader

import myfitnesspal
import browser_cookie3

# ------------------------------------------------------------------------------
# 1) Setup Logging
# ------------------------------------------------------------------------------
LOG_DIR = "/Users/andromeda/Desktop/mfp-health-reports/resources"
LOG_FILE = os.path.join(LOG_DIR, "mfp_report.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------------------------------------------------------------
# 2) User/Global Configuration
# ------------------------------------------------------------------------------
START_DATE = datetime.date(2025, 1, 20)
END_DATE   = datetime.date.today() - datetime.timedelta(days=1)  # "yesterday"
RESOURCES_DIR = "/Users/andromeda/Desktop/mfp-health-reports/resources"
REPORT_TITLE = "MyFitnessPal Summary Report"

# ------------------------------------------------------------------------------
# 3) Color & Styling
# ------------------------------------------------------------------------------
COLOR_WEIGHT_LINE = "#435aa9"
COLOR_CALORIES    = "#435aa9"
COLOR_CARBS       = "#7FB8A5"
COLOR_FAT         = "#E4A96B"
COLOR_PROTEIN     = "#C58CB0"
COLOR_MINUTES     = "#4fa77b"
COLOR_CAL_EXER    = "#cf7b20"
COLOR_WEEKLY_LINE = "#cf1000"
COLOR_LOST        = "#5FBF77"
COLOR_GAINED      = "#d85656"
COLOR_NEUTRAL     = "#1e003e"

LEGEND_FONT_SIZE = 9
LEGEND_LOC       = "upper left"

sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
matplotlib.rc('font', **{'family':'sans-serif','sans-serif':['Lato','Arial','DejaVu Sans']})

# ------------------------------------------------------------------------------
# 4) Fetch Data from MyFitnessPal API
# ------------------------------------------------------------------------------
def fetch_data_api(start_date, end_date):
    logging.info("Fetching data from MyFitnessPal API...")

    cookies = browser_cookie3.chrome(domain_name="myfitnesspal.com")
    client = myfitnesspal.Client(cookiejar=cookies)

    # Get all weight measurements
    weight_measurements = client.get_measurements(lower_bound=start_date, upper_bound=end_date)

    food_data = []
    weight_data_list = []
    exercise_data = []

    delta_days = (end_date - start_date).days + 1

    for day_offset in range(delta_days):
        single_date = start_date + datetime.timedelta(day_offset)
        date_str = single_date.strftime('%Y-%m-%d')

        # Fetch daily nutrition totals
        try:
            day_summary = client.get_date(single_date.year, single_date.month, single_date.day)
            food_totals = day_summary.totals
            food_data.append({
                "Date": date_str,
                "calories": food_totals.get("calories", 0),
                "protein":  food_totals.get("protein", 0),
                "carbs":    food_totals.get("carbohydrates", 0),
                "fat":      food_totals.get("fat", 0),
            })
        except Exception as e:
            logging.warning(f"Error fetching nutrition data for {date_str}: {e}")

        # Fetch weight if available
        if single_date in weight_measurements:
            weight_data_list.append({
                "Date": date_str,
                "weight": weight_measurements[single_date]
            })

        # Fetch exercise data (cardio only, from day_summary.exercises)
        try:
            exercises = day_summary.exercises
            total_calories = 0
            total_minutes = 0
            if len(exercises) > 0:
                for cardio in exercises[0].get_as_list():
                    burned = cardio["nutrition_information"].get("calories burned", 0) or 0
                    mins = cardio["nutrition_information"].get("minutes", 0) or 0
                    total_calories += burned
                    total_minutes += mins
            exercise_data.append({
                "Date": date_str,
                "exercise_cal": total_calories,
                "exercise_minutes": total_minutes,
            })
        except Exception as e:
            logging.warning(f"Error fetching exercise data for {date_str}: {e}")

    # Convert to DataFrames
    df_weight = pd.DataFrame(weight_data_list)
    df_food = pd.DataFrame(food_data)
    df_ex = pd.DataFrame(exercise_data)

    # Log data frame sizes
    logging.info(f"Weight data size: {df_weight.shape}")
    logging.info(f"Nutrition data size: {df_food.shape}")
    logging.info(f"Exercise data size: {df_ex.shape}")

    # Clean up weight DataFrame
    if not df_weight.empty:
        df_weight["Date"] = pd.to_datetime(df_weight["Date"], errors="coerce")
        df_weight.dropna(subset=["Date"], inplace=True)
        df_weight.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        df_weight.sort_values("Date", inplace=True)
    else:
        df_weight = pd.DataFrame(columns=["Date", "weight"])
        logging.warning("Weight data is empty!")

    # Clean up nutrition DataFrame
    if not df_food.empty:
        df_food["Date"] = pd.to_datetime(df_food["Date"], errors="coerce")
        df_food.dropna(subset=["Date"], inplace=True)
        # Group by "Date" but keep it as a column
        df_food = df_food.groupby("Date", as_index=False).agg({
            "calories": "sum",
            "carbs": "sum",
            "fat": "sum",
            "protein": "sum"
        }).sort_values("Date")
        df_food = df_food[df_food["calories"] != 0]
    else:
        df_food = pd.DataFrame(columns=["Date", "calories", "carbs", "fat", "protein"])
        logging.warning("Nutrition data is empty!")

    # Clean up exercise DataFrame
    if not df_ex.empty:
        df_ex["Date"] = pd.to_datetime(df_ex["Date"], errors="coerce")
        df_ex.dropna(subset=["Date"], inplace=True)
        # Group by "Date" but keep it as a column
        df_ex = df_ex.groupby("Date", as_index=False).agg({
            "exercise_cal": "sum",
            "exercise_minutes": "sum"
        }).sort_values("Date")
    else:
        df_ex = pd.DataFrame(columns=["Date", "exercise_cal", "exercise_minutes"])
        logging.warning("Exercise data is empty!")

    return df_weight, df_food, df_ex

# ------------------------------------------------------------------------------
# 5) Computation and Plotting Functions
# ------------------------------------------------------------------------------
def compute_3_day_sliding_average(weight_df):
    if not weight_df.empty and "weight" in weight_df.columns:
        weight_df["weight_3d_avg"] = weight_df["weight"].rolling(window=3).mean()
    return weight_df

def generate_weight_graph(weight_df, output_path):
    if weight_df.empty or "weight_3d_avg" not in weight_df.columns:
        return
    # Sort by Date to ensure a proper time sequence
    sorted_df = weight_df.sort_values("Date")
    plt.figure(figsize=(9, 5))
    plt.plot(sorted_df["Date"], sorted_df["weight_3d_avg"], color=COLOR_WEIGHT_LINE, linewidth=2)
    plt.title("Daily Weight (3-day Avg)", fontsize=14, pad=10)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Weight (lb)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_calories_macros_graph(nutrition_df, output_path):
    if nutrition_df.empty:
        return

    sorted_df = nutrition_df.sort_values("Date")

    # First chart: Daily Calories
    cal_path = output_path.replace(".png", "_calories.png")
    plt.figure(figsize=(9, 5))
    plt.plot(sorted_df["Date"], sorted_df["calories"], 
             label="Calories", color=COLOR_CALORIES, linewidth=2)
    plt.title("Daily Calories", fontsize=14, pad=10)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Calories", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(fontsize=LEGEND_FONT_SIZE, loc=LEGEND_LOC)
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(cal_path, dpi=300)
    plt.close()

    # Second chart: Daily Macros
    mac_path = output_path.replace(".png", "_macros.png")
    plt.figure(figsize=(9, 5))
    plt.plot(sorted_df["Date"], sorted_df["carbs"], 
             label="Carbs (g)", color=COLOR_CARBS, linewidth=2)
    plt.plot(sorted_df["Date"], sorted_df["fat"], 
             label="Fat (g)", color=COLOR_FAT, linewidth=2)
    plt.plot(sorted_df["Date"], sorted_df["protein"], 
             label="Protein (g)", color=COLOR_PROTEIN, linewidth=2)
    plt.title("Daily Macros", fontsize=14, pad=10)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Grams", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(fontsize=LEGEND_FONT_SIZE, loc=LEGEND_LOC)
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(mac_path, dpi=300)
    plt.close()

def generate_minutes_histogram(exercise_df, output_path):
    if exercise_df.empty or "exercise_minutes" not in exercise_df.columns:
        return
    # We need the Date to be an index for resampling
    temp_df = exercise_df.copy()
    temp_df.set_index("Date", inplace=True)
    temp_df.sort_index(inplace=True)

    x_values = temp_df.index.map(lambda d: d.toordinal())
    y_values = temp_df["exercise_minutes"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_values, y_values, width=0.8, color=COLOR_MINUTES, edgecolor=None, alpha=0.9)

    # Weekly average line
    weekly_data = temp_df["exercise_minutes"].resample("W-SUN")
    for week_end, weekly_series in weekly_data:
        if not weekly_series.empty:
            week_avg = weekly_series.mean()
            week_start = week_end - pd.Timedelta(days=6)
            subset = temp_df.index[(temp_df.index >= week_start) & (temp_df.index <= week_end)]
            if len(subset) > 0:
                subset_ordinals = subset.map(lambda d: d.toordinal())
                ax.hlines(y=week_avg,
                          xmin=min(subset_ordinals)-0.4,
                          xmax=max(subset_ordinals)+0.4,
                          color=COLOR_WEEKLY_LINE, linestyles="-", linewidth=2, alpha=0.9)

    ax.set_title("Daily Exercise Minutes", fontsize=14, pad=10)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Minutes", fontsize=12)
    if len(x_values) > 0:
        tick_interval = max(1, len(x_values)//10)
        tick_positions = x_values[::tick_interval]
        # We need the original sorted date index to label
        tick_labels = [temp_df.index[i].strftime("%m/%d") for i in range(0, len(x_values), tick_interval)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def generate_calories_histogram(exercise_df, output_path):
    if exercise_df.empty or "exercise_cal" not in exercise_df.columns:
        return
    temp_df = exercise_df.copy()
    temp_df.set_index("Date", inplace=True)
    temp_df.sort_index(inplace=True)

    x_values = temp_df.index.map(lambda d: d.toordinal())
    y_values = temp_df["exercise_cal"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_values, y_values, width=0.8, color=COLOR_CAL_EXER, edgecolor=None, alpha=0.9)

    # Weekly average line
    weekly_data = temp_df["exercise_cal"].resample("W-SUN")
    for week_end, weekly_series in weekly_data:
        if not weekly_series.empty:
            week_avg = weekly_series.mean()
            week_start = week_end - pd.Timedelta(days=6)
            subset = temp_df.index[(temp_df.index >= week_start) & (temp_df.index <= week_end)]
            if len(subset) > 0:
                subset_ordinals = subset.map(lambda d: d.toordinal())
                ax.hlines(y=week_avg,
                          xmin=min(subset_ordinals)-0.4,
                          xmax=max(subset_ordinals)+0.4,
                          color=COLOR_WEEKLY_LINE, linestyles="-", linewidth=2, alpha=0.9)

    ax.set_title("Daily Exercise Calories", fontsize=14, pad=10)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Calories", fontsize=12)
    if len(x_values) > 0:
        tick_interval = max(1, len(x_values)//10)
        tick_positions = x_values[::tick_interval]
        tick_labels = [temp_df.index[i].strftime("%m/%d") for i in range(0, len(x_values), tick_interval)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def compute_total_weight_lost(weight_df):
    """
    Compare the *first 3-day average* vs. the *last 3-day average*.
    If fewer than 3 data points exist, return 0.0
    """
    if weight_df.empty or len(weight_df) < 3:
        return 0.0

    # Sort by Date to isolate the first 3 days and last 3 days
    sorted_df = weight_df.sort_values("Date")
    if len(sorted_df) < 3:
        return 0.0

    first_3 = sorted_df.iloc[:3]["weight"].mean()
    last_3  = sorted_df.iloc[-3:]["weight"].mean()

    return first_3 - last_3

def compute_last_week_loss(weight_df):
    """
    Compare weight 7 days ago vs. the most recent day.
    Returns None if fewer than 7 days of data.
    """
    if len(weight_df) < 7:
        return None

    sorted_df = weight_df.sort_values("Date")
    last_date = sorted_df["Date"].iloc[-1]
    date_7_days_ago = last_date - pd.Timedelta(days=7)

    prior_data = sorted_df[sorted_df["Date"] <= date_7_days_ago]
    if prior_data.empty:
        return None

    ref_weight  = prior_data["weight"].iloc[-1]
    last_weight = sorted_df["weight"].iloc[-1]

    return ref_weight - last_weight


def estimate_maintenance_calories(weight_df, nutrition_df, intervals=[7, 14, 21, 28, 35]):
    results = []

    if weight_df.empty or nutrition_df.empty:
        return None

    # Ensure Date column exists and is in datetime format
    if "Date" in weight_df.columns:
        weight_df["Date"] = pd.to_datetime(weight_df["Date"], errors="coerce")
    else:
        raise KeyError("The 'Date' column is missing from weight_df.")
    
    if "Date" in nutrition_df.columns:
        nutrition_df["Date"] = pd.to_datetime(nutrition_df["Date"], errors="coerce")
    else:
        raise KeyError("The 'Date' column is missing from nutrition_df.")

    # Now we set the index in this function for the join logic
    weight_df.set_index("Date", inplace=True)
    nutrition_df.set_index("Date", inplace=True)

    # Sort index to ensure chronological order
    weight_df.sort_index(inplace=True)
    nutrition_df.sort_index(inplace=True)

    # Only consider weight data starting from the first day food was tracked
    food_start_date = nutrition_df.index.min()
    filtered_weight_df = weight_df[weight_df.index >= food_start_date]

    # Join the filtered weight data with nutrition data
    combined_df = filtered_weight_df.join(nutrition_df, how="inner")
    combined_df.dropna(subset=["weight", "calories"], inplace=True)
    if len(combined_df) < 2:
        return None

    sorted_dates = combined_df.index.sort_values()

    for interval in intervals:
        for i in range(len(sorted_dates) - interval + 1):
            start_date = sorted_dates[i]
            end_date = sorted_dates[i + interval - 1]

            # Compute weight change
            weight_change = combined_df.loc[end_date, "weight"] - combined_df.loc[start_date, "weight"]
            net_cal_balance = weight_change * 3500

            # Compute average calorie intake over the interval
            avg_cal_eaten = combined_df.loc[start_date:end_date, "calories"].mean()
            net_cal_balance_per_day = net_cal_balance / interval

            maintenance_calories = avg_cal_eaten - net_cal_balance_per_day
            results.append(maintenance_calories)

    # Compute the final averaged maintenance calorie estimate
    return np.mean(results)


def compute_7_day_calorie_averages(nutrition_df):
    """
    Return (avg_last_7_days, overall_avg).
    If fewer than 7 days total, (None, overall_avg) if any data is present.
    """
    if nutrition_df.empty:
        return (None, None)

    # Sort by Date to handle the most recent 7 days
    sorted_df = nutrition_df.sort_values("Date")
    overall_avg = sorted_df["calories"].mean()

    if len(sorted_df) < 7:
        return (None, round(overall_avg, 2))

    last_date = sorted_df["Date"].iloc[-1]
    date_7_days_ago = last_date - pd.Timedelta(days=6)
    recent_7 = sorted_df[sorted_df["Date"] >= date_7_days_ago]

    if recent_7.empty:
        return (None, round(overall_avg, 2))

    return (round(recent_7["calories"].mean(), 2), round(overall_avg, 2))

def color_loss_string(value):
    """
    Return (action_string, value_string, highlight_color).
    """
    if value is None:
        return ("N/A", "N/A", COLOR_NEUTRAL)
    if value > 0:
        # positive => user lost
        return ("lost", f"{abs(value):.2f}", COLOR_LOST)
    elif value < 0:
        # negative => user gained
        return ("gained", f"{abs(value):.2f}", COLOR_GAINED)
    else:
        return ("neither gained nor lost", "0.00", COLOR_NEUTRAL)

# ------------------------------------------------------------------------------
# 6) Main
# ------------------------------------------------------------------------------
def main():
    try:
        # 1. Fetch the data from MyFitnessPal
        weight_df, nutrition_df, exercise_df = fetch_data_api(START_DATE, END_DATE)
        if not weight_df.empty:
            num_days = (weight_df["Date"].max() - weight_df["Date"].min()).days + 1
        else:
            num_days = 0  # If no data, set to 0


        logging.info("Data fetched successfully.")

        # 2. Compute 3-day average for weight
        weight_df = compute_3_day_sliding_average(weight_df)

        # 3. Prepare resources directory
        os.makedirs(RESOURCES_DIR, exist_ok=True)

        # 4. Generate plots
        generate_weight_graph(weight_df, os.path.join(RESOURCES_DIR, "weight_graph.png"))
        generate_calories_macros_graph(nutrition_df, os.path.join(RESOURCES_DIR, "calories_macros.png"))
        generate_minutes_histogram(exercise_df, os.path.join(RESOURCES_DIR, "exercise_summary_minutes.png"))
        generate_calories_histogram(exercise_df, os.path.join(RESOURCES_DIR, "exercise_summary_calories.png"))

        # 5. Compute summary statistics
        since_start_loss = compute_total_weight_lost(weight_df)
        last_week_loss = compute_last_week_loss(weight_df)
        maintenance_est = estimate_maintenance_calories(weight_df.copy(), nutrition_df.copy())
        last_7_avg_cals, overall_avg_cals = compute_7_day_calorie_averages(nutrition_df)
        if maintenance_est and last_7_avg_cals:
            daily_deficit_7d = maintenance_est - last_7_avg_cals
        else:
            daily_deficit_7d = None

        # 6. Render the HTML report
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        env = Environment(loader=FileSystemLoader(RESOURCES_DIR))
        template = env.get_template("report_template.html")

        context = {
            "report_title": REPORT_TITLE,
            "last_updated": now_str,
            "num_days": num_days,
            "start_action": color_loss_string(since_start_loss)[0],
            "start_val": f"{since_start_loss:.2f}",
            "lw_action": color_loss_string(last_week_loss)[0],
            "lw_val": f"{last_week_loss:.2f}",
            "maintenance_est": f"{maintenance_est:.2f}" if maintenance_est else "N/A",
            "last_7_avg_cals": f"{last_7_avg_cals:.2f}" if last_7_avg_cals else "N/A",
            "overall_avg_cals": f"{overall_avg_cals:.2f}" if overall_avg_cals else "N/A",
            "weight_graph_path": "weight_graph.png" if not weight_df.empty else None,
            "exercise_minutes_path": "exercise_summary_minutes.png" if not exercise_df.empty else None,
            "exercise_calories_path": "exercise_summary_calories.png" if not exercise_df.empty else None,
            "calories_graph_path": "calories_macros_calories.png" if not nutrition_df.empty else None,
            "macros_graph_path": "calories_macros_macros.png" if not nutrition_df.empty else None,
            "color_lost": COLOR_LOST,
            "color_gained": COLOR_GAINED,
            "color_neutral": COLOR_NEUTRAL,
            "daily_deficit_7d": f"{daily_deficit_7d:.2f}" if daily_deficit_7d else "N/A"
        }

        html_report_path = "/Users/andromeda/Desktop/mfp-health-reports/report.html"
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write(template.render(context))

        logging.info(f"Report generated at {html_report_path}.")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
    
    logging.info(f"\n")


if __name__ == "__main__":
    main()
