"""
House Price Prediction using Machine Learning

Dataset:
Ames House Prices dataset from OpenML
https://www.openml.org/d/42165

Program flow:
1. Ask permission to download/load the dataset.
2. Ask permission to show total available features.
3. Ask permission to select important features.
4. Ask permission to prepare and split the data.
5. Ask permission to train and tune the model.
6. Ask permission to evaluate the model.
7. Ask permission to predict prices for new unseen houses.

If the user enters anything other than "yes" or "y",
the program exits safely.
"""

import sys
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# FUNCTION: ASK USER WHETHER TO CONTINUE
# ------------------------------------------------------------
def continue_to_next_step(step_name):
    """
    Ask the user whether they want to continue to a step.

    The program proceeds only when the user enters:
    yes  OR  y

    Any other input exits the program.
    """

    print("\n" + "=" * 70)
    user_choice = input(
        f"{step_name}? "
        "(yes/no): "
    ).strip().lower()

    if user_choice in ["yes", "y"]:
        # print(f"\nProceeding to: {step_name}")
        return True

    print("\nProgram stopped by the user.")
    print("Thank you.")
    sys.exit()


# ------------------------------------------------------------
# FUNCTION: GET VALID NUMERIC INPUT
# ------------------------------------------------------------
def get_numeric_input(prompt, minimum=None, maximum=None):
    """
    Continuously asks the user for a valid numeric input.
    """

    while True:
        try:
            value = float(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Please enter a value greater than or equal to {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Please enter a value less than or equal to {maximum}.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a numeric value.")


# ============================================================
# STEP 1: DOWNLOAD AND LOAD DATASET
# ============================================================
continue_to_next_step("Step 1 - Download and load the dataset")

print("\nDownloading/loading the Ames House Prices dataset...\n")

housing = fetch_openml(
    data_id=42165,
    as_frame=True
)

data = housing.data.copy()
target = pd.to_numeric(housing.target, errors="coerce")

print("Dataset loaded successfully.")
print(f"Total house records: {len(data)}")


# ============================================================
# STEP 2: SHOW TOTAL AVAILABLE FEATURES AND DATA PREVIEW
# ============================================================
continue_to_next_step("Step 2 - Show total features and dataset preview")

original_features = data.columns.tolist()

print("\n" + "=" * 70)
print("TOTAL FEATURES AVAILABLE IN THE DATASET")
print("=" * 70)

print(f"\nTotal original input features available: {len(original_features)}")
print("Target variable: SalePrice")

# Create a preview table that also includes the target variable.
dataset_preview = data.copy()
dataset_preview["SalePrice"] = target

print("\nFirst 5 rows of the dataset:\n")
print(dataset_preview.head())


# ============================================================
# STEP 3: FEATURE ENGINEERING AND FEATURE SELECTION
# ============================================================
continue_to_next_step("Step 3 - Create and select important features")

print("\nCreating a new feature: HouseAge")

# HouseAge = year of sale - year of construction.
data["HouseAge"] = (
    pd.to_numeric(data["YrSold"], errors="coerce")
    - pd.to_numeric(data["YearBuilt"], errors="coerce")
)

# Selected important features for the classroom demonstration.
FEATURES = [
    "GrLivArea",       # Living area in square feet
    "LotArea",         # Lot size in square feet
    "BedroomAbvGr",    # Bedrooms above ground
    "FullBath",        # Number of full bathrooms
    "HouseAge",        # Age of house
    "OverallQual"      # Overall quality score: 1 to 10
]

FEATURE_DESCRIPTIONS = {
    "GrLivArea": "Above-ground living area in square feet",
    "LotArea": "Lot area in square feet",
    "BedroomAbvGr": "Number of bedrooms above ground",
    "FullBath": "Number of full bathrooms",
    "HouseAge": "House age in years",
    "OverallQual": "Overall material and finish quality, from 1 to 10"
}

print("\n" + "=" * 70)
print("IMPORTANT FEATURES SELECTED FOR THE MODEL")
print("=" * 70)

print(f"\nNumber of selected important features: {len(FEATURES)}\n")

for index, feature in enumerate(FEATURES, start=1):
    print(f"{index}. {feature}")
    print(f"   Meaning: {FEATURE_DESCRIPTIONS[feature]}")


# ============================================================
# STEP 4: PREPARE AND SPLIT DATA
# ============================================================
continue_to_next_step("Step 4 - Prepare data and split it into training and testing sets")

print("\nPreparing the selected features...")

X = data[FEATURES].copy()

# Convert values into numeric format.
for column in FEATURES:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Keep rows having a valid target value.
valid_rows = target.notna()

X = X.loc[valid_rows]
y = target.loc[valid_rows]

print(f"Records available after data preparation: {len(X)}")

print("\nSplitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ============================================================
# STEP 5: CREATE, TRAIN, AND TUNE MODEL
# ============================================================
continue_to_next_step("Step 5 - Train and tune the Random Forest model")

print("\nCreating the machine-learning pipeline...")

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Different model settings to test during parameter tuning.
parameter_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 15, 25],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

print("\nTraining and tuning the model using 5-fold cross-validation...")
print("Please wait...\n")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Model training completed successfully.\n")

print("Best model parameters found:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")


# ============================================================
# STEP 6: EVALUATE MODEL
# ============================================================
continue_to_next_step("Step 6 - Evaluate the model on unseen test data")

print("\nMaking predictions for unseen test data...")

test_predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
rmse = root_mean_squared_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print("\n" + "=" * 70)
print("MODEL EVALUATION RESULTS")
print("=" * 70)

print(f"\nMean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²): {r2:.3f}")

print("\nInterpretation:")
print(f"- On average, the model prediction differs by about ${mae:,.0f}.")
print("- Lower MAE and RMSE values indicate better prediction performance.")
print("- An R² value closer to 1 indicates a stronger model fit.")


# ============================================================
# STEP 7: PREDICT NEW UNSEEN HOUSE
# ============================================================
continue_to_next_step("Step 7 - Predict the price of a new unseen house")

print("\n" + "=" * 70)
print("HOUSE PRICE PREDICTION FOR NEW UNSEEN EXAMPLES")
print("=" * 70)

while True:
    print("\nEnter the details of a new house:\n")

    living_area = get_numeric_input(
        "Living area in square feet: ",
        minimum=1
    )

    lot_area = get_numeric_input(
        "Lot area in square feet: ",
        minimum=1
    )

    bedrooms = get_numeric_input(
        "Number of bedrooms: ",
        minimum=0
    )

    full_bathrooms = get_numeric_input(
        "Number of full bathrooms: ",
        minimum=0
    )

    house_age = get_numeric_input(
        "Age of house in years: ",
        minimum=0
    )

    overall_quality = get_numeric_input(
        "Overall quality (1 = low, 10 = excellent): ",
        minimum=1,
        maximum=10
    )

    # Create a one-row DataFrame for the new house.
    new_house = pd.DataFrame(
        [[
            living_area,
            lot_area,
            bedrooms,
            full_bathrooms,
            house_age,
            overall_quality
        ]],
        columns=FEATURES
    )

    # Generate a prediction.
    predicted_price = best_model.predict(new_house)[0]

    print("\n" + "-" * 70)
    print(f"Predicted house sale price: ${predicted_price:,.2f}")
    print("-" * 70)

    predict_again = input(
        "\nDo you want to predict another unseen house? "
        "(yes/no): "
    ).strip().lower()

    if predict_again not in ["yes", "y"]:
        print("\nPrediction session ended successfully.")
        print("Thank you.")
        break