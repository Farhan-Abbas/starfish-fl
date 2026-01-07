# Task Configuration Guide for Starfish Controller

## Understanding the Tasks Field

When creating a new project, the **Tasks** field requires a JSON array that defines the federated learning workflow. Each task represents a step in the training process.

## Task Structure

Each task must have:
- **seq**: Sequential number (starting from 1)
- **model**: The ML model class name (e.g., "LogisticRegression")
- **config**: Configuration dictionary for the task

## Valid Task Examples

### Example 1: Single Task (Logistic Regression)

```json
[
  {
    "seq": 1,
    "model": "LogisticRegression",
    "config": {
      "total_round": 5,
      "current_round": 1,
      "local_epochs": 1,
      "learning_rate": 0.01
    }
  }
]
```

### Example 2: Multiple Sequential Tasks

```json
[
  {
    "seq": 1,
    "model": "LogisticRegression",
    "config": {
      "total_round": 3,
      "current_round": 1,
      "local_epochs": 1
    }
  },
  {
    "seq": 2,
    "model": "LogisticRegression",
    "config": {
      "total_round": 5,
      "current_round": 1,
      "local_epochs": 2
    }
  }
]
```

## Validation Rules

The system validates:

1. **At least one task** must be provided
2. **seq must start at 1** and be consecutive (1, 2, 3...)
3. **Each task must have** `seq`, `model`, and `config` keys
4. **seq must be** a non-negative integer
5. **model must exist** in `starfish/controller/tasks/`
6. **config must be** a non-empty dictionary

## Currently Available Models

### Logistic Regression

**Description:** Binary classification using logistic regression

**Use Case:** Predicting binary outcomes (Yes/No, 0/1, True/False)

**File Location:** `starfish/controller/tasks/logistic_regression.py`

**Dataset Requirements:** CSV with features in all columns except last, binary label (0 or 1) in last column

### Linear Regression

**Description:** Continuous value prediction using linear regression

**Use Case:** Predicting continuous numerical outcomes (e.g., prices, temperatures, life expectancy)

**File Location:** `starfish/controller/tasks/linear_regression.py`

**Dataset Requirements:** CSV with features in all columns except last, continuous target value in last column

### Statistical Logistic Regression

**Description:** Statistical logistic regression with inference outputs (coefficients, p-values, confidence intervals)

**Use Case:** Binary classification with focus on statistical significance

**File Location:** `starfish/controller/tasks/stats_models/logistic_regression_stats.py`

**Dataset Requirements:** CSV with features in all columns except last, binary outcome (0 or 1) in last column. Minimum 30 samples required.

**Statistical Outputs:**
- Coefficients (Log-Odds) with standard errors, p-values, confidence intervals
- Odds Ratios (exponentiated coefficients)
- Pseudo R-squared (McFadden's)
- Likelihood Ratio Chi-Squared statistic

### ANCOVA

**Description:** Analysis of Covariance - tests group differences while controlling for continuous covariates

**Use Case:** Comparing group means while accounting for continuous variables (e.g., treatment effects controlling for age)

**File Location:** `starfish/controller/tasks/stats_models/ancova.py`

**Dataset Requirements:** CSV with:
- First K columns: group indicators (one-hot encoded)
- Middle columns: continuous covariates
- Last column: continuous outcome variable

Minimum 30 samples required.

**Statistical Outputs:**
- Coefficients with standard errors, p-values, confidence intervals
- F-statistics for group effects
- Partial eta-squared (effect size)
- Adjusted group means

## Configuration Parameters Explained

### Required Parameters

- **total_round**: Total number of federated learning rounds (how many times models are aggregated)
- **current_round**: The round to start with (in most cases we start with 1)

### Optional Parameters

- **local_epochs**: Number of epochs each site trains locally (default: 1 in code)
- **test_size**: Proportion of data used for testing (default: 0.2 in code)
- **learning_rate**: Learning rate for training (if applicable)
- **n_group_columns**: (Ancova only) Number of columns representing group membership
- **description**: Optional description of what this task does

## Step-by-Step: Creating a Project

1. **Go to**: http://localhost:8001/controller/projects/new/
2. **Project Name**: Enter "Test Project"
3. **Project Description**: Enter "Testing federated learning"
4. **Tasks**: Copy and paste this:
   ```json
   [{"seq":1,"model":"LogisticRegression","config":{"total_round":5,"current_round":1}}]
   ```
5. **Click Submit**
