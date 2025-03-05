# MyFitnessPal Health Reports

## Overview
This project automates the retrieval, analysis, and visualization of health and fitness data from MyFitnessPal. The script fetches user data such as weight, calorie intake, macronutrients, and exercise activity, then generates summary statistics and visual reports.

## Features
- Fetches data from MyFitnessPal via its API
- Logs data collection process
- Computes weight trends and calorie averages
- Generates visual reports using Matplotlib and Seaborn
- Outputs an HTML report summarizing the data
- Runs daily at 6 AM via macOS LaunchAgent

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install myfitnesspal browser-cookie3 pandas numpy matplotlib seaborn jinja2
```

## Installation
Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/your-repo/mfp-health-reports.git
cd mfp-health-reports
```

Set up the required directory structure:

```sh
mkdir -p resources
```

## Usage
Run the script manually:

```sh
python3 my_script.py
```

The script fetches data, generates reports, and outputs an HTML file in the `resources` directory.

## Automated Execution (macOS LaunchAgent)
The script is scheduled to run daily at 6 AM using a LaunchAgent.

To load the LaunchAgent:

```sh
cp com.andromeda.mfphealthreports.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.andromeda.mfphealthreports.plist
```

To check its status:

```sh
launchctl list | grep com.andromeda.mfphealthreports
```

To unload it:

```sh
launchctl unload ~/Library/LaunchAgents/com.andromeda.mfphealthreports.plist
```

## Outputs
- **Logs**: Stored in `resources/mfp_report.log`
- **HTML Report**: Generated as `report.html` in `resources/`
- **Graphs**: Stored in `resources/` (e.g., `weight_graph.png`, `calories_macros.png`)

## Customization
Modify the configuration in `my_script.py`:
- `START_DATE` and `END_DATE` to adjust the analysis period
- Colors and styling under "Color & Styling"
- `REPORT_TITLE` to customize the report name

## Troubleshooting
- If the script fails, check `error.log` in the project directory.
- Ensure MyFitnessPal authentication via browser cookies is working.
- Verify dependencies are installed properly.