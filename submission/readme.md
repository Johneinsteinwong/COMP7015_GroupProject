# Project Submission

This folder contains the submission for the COMP7015 Group Project (Optional 1).

## Installation

To set up the conda environment for this project, follow the steps below:


1. **Create the conda environment:**

    Install the conda environment using the following command:

    ```bash
    conda env create -f environment.yaml
    ```

**Environment Details:**

- **Python Version:** 3.10.15
- **Package Versions:** 

      - bayesian-optimization==2.0.0
      - imbalanced-learn==0.12.4
      - imblearn==0.0
      - joblib==1.4.2
      - lightgbm==4.5.0
      - matplotlib==3.9.2
      - notebook==7.2.2
      - numpy==1.26.4
      - pandas==2.2.3
      - scipy==1.12.0
      - seaborn==0.13.2
      - tabulate==0.9.0

2. **Activate the conda environment:**

    ```bash
    conda activate comp7015
    ```
## Usage

After setting up the environment, you can run the notebooks in the following orders:

1. **Exploratory data analysis and feature engineering**

- This notebook will give you insight about the data and show you the steps to preprocess the data and compute the features.
```bash
1_Feature Engineering (Isolation Forest with Parameter Tuning).ipynb
```
2. **Hyperparameter search**
- This notebook searches for optimal values of hyperparameters for all models (probit, logistic, LightGBM) and save the hyperparameters in the json files of corresponding models (probit.json, logistic.json, lightgbm.json).
```bash
2_hyperparameter_search.ipynb
```
3. **Probit and Logistic models**
- This notebook trains the probit and logistic models using the hyperparameters found in step 2 and all the data, and save the data preprocessing (sklearn) pipeline to `*_pipeline.pkl` as well as the models to `probit.pkl` and `logistic.pkl`. 
- After training, a stratified 5-fold cross validation will be performed to estimate the models' performance, the summaries will be save in the folders `probit` and `logistic` respectively.
```bash
3_train_probit_logistic.ipynb
```

4. **LightGBM model**
- This notebook trains the LightGBM model using the hyperparameters found in step 2 and all the data, and save the data preprocessing (sklearn) pipeline to `lightgbm_pipeline.pkl` as well as the model to `lightgbm.txt`. 
- After training, a stratified 5-fold cross validation will be performed to estimate the model's performance, the summary will be save in the folder `lightgbm`.
```bash
4_train_lightgbm.ipynb
```

5. **Making a prediction**
- This notebook will generate predictions made by the LightGBM model (or also the probit and logistic models).
```bash
5_make_prediction.ipynb
```