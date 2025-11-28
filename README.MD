# 2025 FALL DATA SCIENCE FINAL PROJECT

## Santander Product Recommendation System

This project builds a web application that recommends bank products to customers based on their profile using machine learning.

## Project Structure
```
Final/
├── Data Science Term-Project Rules.pdf
├── README.MD
├── dataset/
│   ├── sample_submission.csv
│   ├── test_ver2.csv
│   └── train_ver2.csv
├── group5_proposal.pdf
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
├── requirements.txt
├── src/
│   ├── preprocessor.py
│   ├── train_model.py
│   └── utils/
│       └── const.py
├── test/
│   ├── check_data.py
│   └── test_prediction.py
└── web/
	├── app.py
	└── templates/
		├── form_page.html
		├── layout.html
		└── submit.html
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time Only)
Before running the web application, you need to train the prediction model:

```bash
python train_model.py
```

This will:
- Load the training data from `dataset/train_ver2.csv`
- Train a machine learning model
- Save the model and preprocessor to `models/` directory

**Note:** Training may take several minutes depending on your computer and sample size.

### 3. Run the Web Application
```bash
python web/app.py
```

Then open your browser and go to: `http://127.0.0.1:5000`

## How It Works

1. **User Input**: The web form collects 24 customer features (age, income, account type, etc.)
2. **Prediction**: The trained model predicts the probability of purchasing each of 24 bank products
3. **Recommendations**: Top 7 products are recommended based on highest probabilities
4. **Display**: Results are shown with probabilities and visual indicators

## Features

- **Multi-step Form**: User-friendly interface with step-by-step data collection
- **Real-time Prediction**: Instant product recommendations using trained ML model
- **Visual Dashboard**: Clean display of customer info and recommended products
- **Product Coverage**: Predicts 24 different banking products

## Model Information

- **Features**: 24 customer attributes (demographics, account info, behavior)
- **Targets**: 24 binary indicators for different bank products
- **Algorithm**: Random Forest with Multi-output Classification
- **Preprocessing**: Automatic handling of categorical/numerical features

## Customization

You can customize the model by editing `train_model.py`:
- Change `sample_size` to use more/less training data
- Modify model parameters (n_estimators, max_depth, etc.)
- Try different algorithms (XGBoost, Neural Networks, etc.)

## Products Being Predicted

1. Saving Account
2. Guarantees
3. Current Accounts
4. Derivada Account
5. Payroll Account
6. Junior Account
7. Más particular Account
8. particular Account
9. particular Plus Account
10. Short-term deposits
11. Medium-term deposits
12. Long-term deposits
13. e-account
14. Funds
15. Mortgage
16. Pensions
17. Loans
18. Taxes
19. Credit Card
20. Securities
21. Home Account
22. Payroll
23. Pensions
24. Direct Debit