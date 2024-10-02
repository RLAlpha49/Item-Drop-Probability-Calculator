
# Item Drop Probability Calculator

## Overview

This project is a Streamlit web application that calculates and visualizes item drop probabilities for games or simulations. It allows users to input parameters such as success probability, desired number of successes, and optional mission time to generate probability tables and graphs.

## Key Features

1. Probability calculation based on user inputs
2. Interactive display of probabilities for different thresholds
3. Visualization of probability distribution
4. Excel file export of detailed results
5. Optional mission time consideration

## File Structure

- `app.py`: Main Streamlit application file
- `pyproject.toml`: Project configuration and dependencies
- `scripts.py`: Utility scripts for project maintenance
- `.github/`: GitHub Actions workflows for CI/CD

## Setup and Installation

1. Ensure Python 3.12 or higher is installed
2. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

## Development

- Use `ruff` for linting and formatting
- Run `mypy` for type checking
- Use the `scripts.py` file for project cleaning and maintenance

## CI/CD

The project uses GitHub Actions for:
- Ruff linting and formatting checks
- CodeQL analysis
- Dependabot updates with auto-merge capability

## Dependencies

Main dependencies include:
- Streamlit
- SciPy
- Matplotlib
- XlsxWriter

For a full list, refer to the `pyproject.toml` file.
