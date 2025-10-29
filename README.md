# Formula 1 Race Prediction Model

Machine learning model predicting Formula 1 race outcomes using historical racing data, driver statistics, and performance metrics.

## Overview

This project develops predictive models for Formula 1 race results by analyzing historical race data, driver performance, team statistics, and qualifying positions. The model uses ensemble learning methods to achieve competitive prediction accuracy for race finishing positions.

## Features

- **Comprehensive Data Analysis**: Analysis of 10+ years of Formula 1 race data including driver performance, team statistics, and qualifying results
- **Feature Engineering**: Derived features including driver consistency metrics, team performance trends, qualifying-to-race conversion rates, and historical finishing positions
- **Multiple ML Algorithms**: Implementation and comparison of Random Forest, Gradient Boosting, and other ensemble methods
- **Data Visualization**: Performance analysis charts, feature importance plots, correlation matrices, and prediction accuracy metrics
- **Model Evaluation**: Cross-validation, confusion matrices, accuracy metrics, and performance comparisons across different seasons

## Technologies

- **Python 3.x**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Random Forest, Gradient Boosting, ensemble methods)
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebook

## Dataset

The model uses historical Formula 1 data including:
- **Race Results**: Finishing positions, lap times, points scored
- **Driver Statistics**: Career wins, podiums, championship standings, DNF rates
- **Team Performance**: Constructor standings, historical performance, reliability metrics
- **Qualifying Data**: Grid positions, qualifying gaps, sector times
- **Track Characteristics**: Circuit type, weather conditions, historical patterns

**Data Sources:**
- Ergast F1 API (historical race data)
- Official F1 statistics
- Historical race results and driver standings

## Model Performance

- **Primary Model**: Gradient Boosting Classifier
- **Prediction Accuracy**: ~72% for race finishing positions
- **Key Predictive Features**: 
  - Qualifying position (strongest predictor)
  - Driver historical performance at specific circuits
  - Team constructor standings
  - Recent race form and consistency
  - Weather conditions and track characteristics

## Installation & Usage

### Prerequisites

```bash
# Python 3.7 or higher required
python --version
```

### Setup

```bash
# Clone repository
git clone https://github.com/Jeremyajay/Formula-1-Prediction-Model.git
cd Formula-1-Prediction-Model

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook cuthbert-f1.ipynb
```

### Running the Model

1. **Data Loading**: Load historical F1 race data from CSV files or API
2. **Preprocessing**: Clean data, handle missing values, encode categorical features
3. **Feature Engineering**: Create derived features for model input
4. **Model Training**: Train Random Forest and Gradient Boosting classifiers
5. **Evaluation**: Analyze prediction accuracy, feature importance, and model performance
6. **Prediction**: Generate predictions for upcoming races or test data

## Project Structure

```
Formula-1-Prediction-Model/
├── cuthbert-f1.ipynb          # Main Jupyter notebook with analysis
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
└── data/                       # Data directory (not tracked in git)
    ├── races.csv               # Historical race results
    └── drivers.csv             # Driver statistics
```

## Key Insights

### Feature Importance Analysis
The model identified the following as most predictive of race outcomes:
1. **Qualifying Position**: Strongest single predictor of race result
2. **Driver Historical Performance**: Win rate and consistency at specific circuits
3. **Team Constructor Standing**: Overall team competitiveness
4. **Recent Form**: Performance in last 3-5 races
5. **Track-Specific Factors**: Circuit characteristics and historical patterns

### Model Comparison
- **Random Forest**: 70% accuracy, good baseline performance
- **Gradient Boosting**: 72% accuracy, best overall performance
- **Logistic Regression**: 65% accuracy, useful for comparison

## Methodology

### Data Preprocessing
- Handled missing values using forward-fill for DNF data
- Normalized numerical features (lap times, points)
- Encoded categorical variables (driver names, team names, circuit names)
- Created training/test split (80/20) with temporal separation

### Feature Engineering
- **Driver Metrics**: Win rate, podium percentage, average finishing position
- **Team Metrics**: Constructor championship points, reliability index
- **Race Context**: Starting grid position, weather conditions, safety car frequency
- **Historical Performance**: Circuit-specific driver performance, head-to-head records

### Model Training
- Cross-validation with 5 folds for robust evaluation
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis to identify key predictors
- Ensemble methods to improve prediction stability

## Challenges & Solutions

**Challenge**: Imbalanced dataset (some drivers have significantly more races)
**Solution**: Applied SMOTE (Synthetic Minority Over-sampling) for class balancing

**Challenge**: High cardinality in driver/team features
**Solution**: Used target encoding based on historical win rates

**Challenge**: Temporal data leakage risk
**Solution**: Ensured strict temporal split - training only on past races

## Future Enhancements

- [ ] **Real-time Predictions**: Integrate with live F1 timing data
- [ ] **Deep Learning**: Experiment with LSTM networks for sequential race data
- [ ] **Telemetry Data**: Incorporate car telemetry (speed, tire wear, fuel load)
- [ ] **Pit Stop Strategy**: Model pit stop timing and tire strategy impacts
- [ ] **Weather Integration**: Real-time weather forecasting for race predictions
- [ ] **Web Interface**: Build Flask/Django app for interactive predictions
- [ ] **Extended Predictions**: Predict qualifying results, fastest laps, DNFs

## Results & Visualizations

The notebook includes:
- Feature correlation heatmaps
- Model accuracy comparison charts
- Confusion matrices for classification performance
- Feature importance bar plots
- Historical accuracy trends across seasons
- Prediction confidence intervals

## Course Context

Developed as part of machine learning coursework at Portland State University, demonstrating:
- End-to-end machine learning pipeline development
- Data collection, cleaning, and preprocessing
- Feature engineering for time-series sports data
- Supervised learning algorithm implementation and comparison
- Model evaluation, validation, and hyperparameter tuning
- Data visualization and results interpretation

## Technical Skills Demonstrated

- **Data Science**: Exploratory data analysis, statistical analysis, feature engineering
- **Machine Learning**: Classification algorithms, ensemble methods, model evaluation
- **Python Programming**: Pandas data manipulation, NumPy numerical computing
- **Visualization**: Matplotlib, Seaborn for publication-quality plots
- **Model Development**: Scikit-learn pipelines, cross-validation, hyperparameter tuning

## Limitations & Considerations

- **Prediction Scope**: Model predicts finishing positions, not exact lap times or margins
- **External Factors**: Cannot account for unexpected events (crashes, mechanical failures, team orders)
- **Data Recency**: Performance may degrade as F1 regulations change significantly
- **Sample Size**: Limited by number of races per season (~23 races/year)
- **Driver Changes**: Model requires retraining when drivers switch teams mid-season

## License

Educational project developed for academic purposes at Portland State University.

## Author

**Jeremy Cuthbert**  
Computer Science Student - Portland State University

## Acknowledgments

- **Data Source**: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?resource=download 
- **Course**: Data with Python coursework at Portland State University
- **Inspiration**: Passion for Formula 1 racing and data-driven sports analytics

---

*Last Updated: August 2025*
