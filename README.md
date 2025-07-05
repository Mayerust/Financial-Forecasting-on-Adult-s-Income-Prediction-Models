# SMARTINTERNZ - Financial Forecasting on Adult’s Income Prediction Models

**Team ID**: SWTID1749710097  
**Team Members**:
-  Mayank Verma (LEAD)
-  Manas Pipersaniya
-  Sanjoli Gupta
-  Sanjog Gupta

   ## PROJECT DEMO VIDEO LINK:
   https://drive.google.com/file/d/1OI9OlIgY_2yOKZ5yL23lavhRmHqyUygQ/view?usp=drive_link

---

## Project Overview
This project predicts income brackets (≤$50k or >$50k) using demographic/occupational data to enable personalized financial planning. It includes:
- Machine learning model training with AdaBoost
- Flask web deployment for real-time predictions
- Three practical business scenarios (investment recommendations, targeted marketing, financial planning optimization)
## Key Features
- Data Preprocessing: Handles categorical encoding, SMOTE balancing, and outlier treatment
- Model Training: Compares Logistic Regression, Decision Trees, Random Forest, and AdaBoost
- Web Interface: User input forms for prediction with result explanations
- Deployment: Flask-based app with modular HTML templates

## Technology Stack
| Component | Technologies/Packages | 
| Backend | Python 3.8+, Flask, scikit-learn | 
| Machine Learning | pandas, numpy, imbalanced-learn | 
| Frontend | HTML/CSS (no JS frameworks) | 
| Model Persistence | joblib | 

---

## Installation & Setup
- Clone Repository:
git clone https://github.com//income-prediction.git
cd income-prediction
- Install Dependencies:
pip install -r requirements.txt  # Sample requirements:
# flask==2.3.2
# scikit-learn==1.2.2
# pandas==2.0.3
# imbalanced-learn==0.10.1
- Download Dataset:
- Get adult.csv from Kaggle
- Place in project root directory
- Train Model:
python train_model.py  # Generates model.joblib

---

## Folder Structure
```
income-prediction/
├── FLASK/
│   ├── templates/              # HTML pages
│   │   ├── index.html          # Homepage
│   │   ├── predict.html        # Input form
│   │   └── result.html         # Prediction results
│   ├── model.joblib            # Trained model
│   └── app.py                  # Flask application
├── train_model.py              # Model training script
├── adult.csv                   # Dataset (not included in repo)
└── requirements.txt            # Dependencies
```

## Usage Instructions
- Start Flask App:
cd FLASK
python app.py
- Access at http://localhost:4000
- Web Interface Workflow:
- Homepage: Project description and navigation
- Predict Page: Submit demographic/occupational details
![Predict Form](https://via.placeholder.com/600x400?text=Predict Income prediction + investment guidance
![Result Example](https://via.placeholder.com/600x200?text=Prediction+Resultmatic Prediction**:
import joblib
model = joblib.load('FLASK/model.joblib')
sample_data = [[40,4,11,2,6,0,4,1,0,0,40,39]]  # Feature vector
print(model.predict(sample_data))  # [0] = ≤50k, [1] = >50k



Model Performance
| Metric | Value | 
| Accuracy | 84.0% | 
| Precision | 72.9% | 
| Recall | 48.6% | 
| F1-Score | 58.2% | 

---

## Data Sources
- Primary Dataset: UCI Adult Census Income
- 32,537 records × 15 features (age, workclass, education, occupation, etc.)
- Preprocessing:
- Replaced ? values with Others
- Label-encoded all categorical variables
- Balanced classes using SMOTE

## Contribution Guidelines
- Reporting Issues:
- Use GitHub Issues for bugs/feature requests
- Development Workflow:
git checkout -b feature/new-algorithm
# Implement changes
pytest tests/  # Add tests for new code
git push origin feature/new-algorithm
- Testing Requirements:
- Maintain ≥80% test coverage
- Validate model accuracy doesn’t drop >2%

---

## License
MIT License - Open for academic/commercial use with attribution.
Note: Dataset licensing follows Kaggle's Terms.

[1] https://medium.datadriveninvestor.com/how-to-write-a-good-readme-for-your-data-science-project-on-github-ebb023d4a50e?gi=8a4060ab4107 [2] https://medium.datadriveninvestor.com/how-to-write-a-good-readme-for-your-data-science-project-on-github-ebb023d4a50e [3] https://dev.to/yuridevat/how-to-create-a-good-readmemd-file-4pa2 [4] https://www.reddit.com/r/AI_Agents/comments/1iix4k8/i_built_an_ai_agent_that_creates_readme_file_for/ [5] https://docsbot.ai/prompts/writing/flask-app-readme-writer [6] https://hackernoon.com/how-to-create-an-engaging-readme-for-your-data-science-project-on-github [7] https://www.reddit.com/r/cscareerquestions/comments/h17blk/always_write_a_clear_readme_if_you_want_to_find_a/ [8] https://www.toolify.ai/ai-news/build-an-income-prediction-model-using-machine-learning-in-python-1747803 [9] https://infoscience.epfl.ch/nanna/record/298249/files/2022_rdm_readme_best_practices.pdf?withWatermark=0&withMetadata=0&version=1&registerDownload=1 [10] https://dev.to/ikhaledabdelfattah/level-up-your-readme-file-495f
