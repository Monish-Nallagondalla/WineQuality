# Wine Quality Prediction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: GPL](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Dagshub](https://img.shields.io/badge/Dagshub-MLflow%20Tracking-blue)](https://dagshub.com/Monish-Nallagondalla/WineQuality)

This is an end-to-end data science project to predict the quality of red wine based on physicochemical properties using ElasticNet regression. The project leverages a modular ML pipeline with **MLflow** for experiment tracking, **Dagshub** for version control and collaboration, and a **Flask**-based web application for user interaction. The pipeline includes data ingestion, validation, transformation, model training, and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Workflows](#workflows)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Details](#model-details)
- [MLflow and Dagshub Integration](#mlflow-and-dagshub-integration)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview
The Wine Quality Prediction project predicts wine quality (on a scale of 0-10) based on features like fixed acidity, volatile acidity, citric acid, and alcohol content. The project uses a modular pipeline implemented in Python, with ElasticNet regression as the core model. It integrates **MLflow** for experiment tracking, **Dagshub** for repository management and experiment visualization, and **Flask** for a web interface to input features and view predictions.

## Dataset
The dataset used is the Wine Quality dataset (`winequality-red.csv`), containing 1599 samples of red wine with the following features:
- **fixed acidity**: Fixed acidity level (g/L)
- **volatile acidity**: Volatile acidity level (g/L)
- **citric acid**: Citric acid level (g/L)
- **residual sugar**: Residual sugar level (g/L)
- **chlorides**: Chloride level (g/L)
- **free sulfur dioxide**: Free sulfur dioxide (mg/L)
- **total sulfur dioxide**: Total sulfur dioxide (mg/L)
- **density**: Density (g/cm³)
- **pH**: pH level
- **sulphates**: Sulphate level (g/L)
- **alcohol**: Alcohol content (% vol)
- **quality**: Wine quality score (0-10, target variable)

**Example Data Row**:
```
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5
```

## ML Pipeline
The project follows a structured ML pipeline:
1. **Data Ingestion**: Loads and extracts the wine quality dataset (`winequality-red.csv`) from `data.zip`.
2. **Data Validation**: Validates the dataset against the schema defined in `schema.yaml`.
3. **Data Transformation**: Performs feature engineering and preprocessing, splitting data into `train.csv` and `test.csv`.
4. **Model Trainer**: Trains an ElasticNet regression model with hyperparameters from `params.yaml`.
5. **Model Evaluation**: Evaluates the model using metrics (MAE, RMSE, R²) tracked via **MLflow** and stored in `metrics.json`.

## Workflows
To update or extend the project, follow these steps:
1. Update `config.yaml` with new configuration settings.
2. Update `schema.yaml` to reflect changes in data schema.
3. Update `params.yaml` with new model hyperparameters.
4. Update the entity definitions in `src/datascience/entity/`.
5. Update the configuration manager in `src/datascience/config/configuration.py`.
6. Update pipeline components in `src/datascience/components/`.
7. Update pipeline scripts in `src/datascience/pipeline/`.
8. Update `main.py` to integrate changes into the main execution flow.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Monish-Nallagondalla/WineQuality.git
   cd WineQuality
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -p venv python=3.8
   conda activate venv/
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Build and run the Docker container:
   ```bash
   docker build -t winequality .
   docker run -p 5000:5000 winequality
   ```

## Usage
1. **Run the ML Pipeline**:
   Execute the main script to run the entire pipeline:
   ```bash
   python main.py
   ```

2. **Run the Flask Web Application**:
   Start the Flask app for predictions via a web interface:
   ```bash
   python app.py
   ```
   Access the application at `http://localhost:5000`.

3. **Explore Research Notebooks**:
   Use Jupyter notebooks in the `research/` directory for detailed analysis:
   ```bash
   jupyter notebook research/
   ```

4. **MLflow Tracking**:
   View experiment logs and metrics:
   ```bash
   mlflow ui
   ```
   Access the MLflow dashboard at `http://localhost:5000`. For Dagshub, experiments are synced to the Dagshub repository.

## File Structure
```
.
├── .gitignore                  # Git ignore file
├── LICENSE                     # GPL License
├── Dockerfile                  # Docker configuration
├── app.py                      # Flask web application
├── main.py                     # Main script to run the pipeline
├── params.yaml                 # Model hyperparameters
├── requirements.txt            # Python dependencies
├── schema.yaml                 # Data schema for validation
├── setup.py                    # Setup script for package installation
├── template.py                 # Template for project setup
├── artifacts/                  # Directory for pipeline outputs
│   ├── data_ingestion/         # Raw and downloaded data
│   │   ├── data.zip
│   │   └── winequality-red.csv
│   ├── data_transformation/    # Transformed datasets
│   │   ├── test.csv
│   │   └── train.csv
│   ├── data_validation/        # Validation status
│   │   └── status.txt
│   ├── model_evaluation/       # Model evaluation metrics
│   │   └── metrics.json
│   └── model_trainer/          # Trained model
│       └── model.joblib
├── config/                     # Configuration files
│   └── config.yaml
├── logs/                       # Log files
│   └── logging.log
├── mlruns/                     # MLflow experiment tracking
│   └── 0/...
├── research/                   # Jupyter notebooks for research
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_validation.ipynb
│   ├── 3_data_transformation.ipynb
│   ├── 4_model_trainer.ipynb
│   ├── 5_model_evaluation.ipynb
│   └── research.ipynb
├── src/datascience/            # Source code for pipeline
│   ├── components/             # Pipeline components
│   ├── config/                 # Configuration management
│   ├── constants/              # Constants
│   ├── entity/                 # Data entities
│   ├── pipeline/               # Pipeline scripts
│   └── utils/                  # Utility functions
├── templates/                  # HTML templates for Flask app
│   ├── index.html
│   └── results.html
```

## Model Details
The project uses **ElasticNet regression**, combining L1 (Lasso) and L2 (Ridge) regularization. Hyperparameters are defined in `params.yaml`:
- **alpha**: Regularization strength
- **l1_ratio**: Balance between L1 and L2 regularization

Model performance metrics (MAE, RMSE, R²) are stored in `artifacts/model_evaluation/metrics.json` and tracked via **MLflow**. The trained model is saved as `model.joblib` in `artifacts/model_trainer/`.

## MLflow and Dagshub Integration
- **MLflow**: Used for experiment tracking, logging metrics (MAE, RMSE, R²), parameters, and models. The `mlruns/` directory contains experiment data, and runs are synced with **Dagshub** for remote tracking.
- **Dagshub**: Hosts the repository at [https://dagshub.com/Monish-Nallagondalla/WineQuality](https://dagshub.com/Monish-Nallagondalla/WineQuality) and provides a platform for visualizing MLflow experiments, collaborating, and versioning datasets and models.

To set up Dagshub integration:
1. Configure MLflow to use Dagshub as the tracking server (set in `config.yaml`).
2. Push experiments to Dagshub using MLflow's tracking URI.

## License
This project is licensed under the **GNU General Public License (GPL)**. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository on GitHub or Dagshub.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request on GitHub or Dagshub.

Ensure code follows the project's standards and includes tests. Update relevant notebooks or pipeline components as needed.

## Contact
For questions or feedback, contact [Monish Nallagondalla](https://github.com/Monish-Nallagondalla) or open an issue on the [GitHub](https://github.com/Monish-Nallagondalla/WineQuality) or [Dagshub](https://dagshub.com/Monish-Nallagondalla/WineQuality) repository.
