# Item Drop Probability Calculator

## Overview

This project is a Streamlit web application that calculates and visualizes item drop probabilities for games or simulations. It allows users to input parameters such as success probability, desired number of successes, and optional mission time to generate probability tables and graphs.

## Key Features

1. **Probability Calculations**
   - Calculate success probabilities based on user-defined parameters
   - Support for multiple attempts and success thresholds
   - Optional mission time constraints

2. **Data Visualization**
   - Interactive probability distribution graphs
   - Detailed probability tables
   - Customizable probability thresholds display

3. **Export Capabilities**
   - Generate Excel files with detailed probability data
   - Export both summary and detailed results

4. **Time Management**
   - Mission time consideration in calculations
   - Time formatting and validation
   - Maximum attempts based on time constraints

## Technical Details

### Main Components

- `format_time()`: Handles time formatting and display
- `calculate_probabilities()`: Core probability calculation engine
- `generate_excel_file()`: Handles data export to Excel
- `render_item_results()`: Manages result visualization

### Project Structure

```
.
├── app.py              # Main Streamlit application
├── scripts.py          # Utility and maintenance scripts
├── pyproject.toml      # Project configuration
└── .github/            # CI/CD workflows
```

## Setup and Installation

1. Ensure Python 3.12 or higher is installed
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Usage

Launch the application:
```bash
streamlit run app.py
```

## Development

### Tools and Standards

- **Code Quality**
  - Ruff for linting and formatting
  - MyPy for type checking
  - Built-in cleaning scripts via `scripts.py`

- **CI/CD Pipeline**
  - GitHub Actions for automated workflows
  - CodeQL security analysis
  - Dependabot integration with auto-merge capability

### Maintenance

Use the built-in cleaning script:
```bash
python scripts.py clean
```

## Dependencies

Core dependencies:
- Streamlit: Web application framework
- SciPy: Statistical calculations
- XlsxWriter: Excel file generation

For a complete list of dependencies, refer to `pyproject.toml`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
