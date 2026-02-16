import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import joblib
import sys
from tqdm import tqdm

#file paths

file_path = "/group/moniergrp/dbaral/run_project/input_data/model_input"
save_path = "/group/moniergrp/dbaral/run_project/output_data/historical_model"

# Get trial number from HPC job array
#trial is the ID of this job in HPC job array
# for ex. if we submit 100 jobs in SLURM, trial = 1 for the first job and so on
#each trial represents - one full modeling on the same dataset but with a different random seed for reproducibility 
#inside each trial, we will still run 100 diff 70/30 splits

#right now we are using trial = 1 becuase we are just using one dataset
#In the future , we will run multiple trials for 
    #13 diff climate models * 2 scenarios = 26 trials
    #each trials will give a separate pipeline and validation results

try:
    trial = int(sys.argv[1])
except IndexError:
    raise ValueError("Pass trial as argument: python lasso_model_job_array.py $SLURM_ARRAY_TASK_ID")

#helper functions

def load_and_prepare_data():
    """
    This function loads the raw dataset and constructs all features.
    This stpe is done once per trial
    It does not depend on random splits or lasso.
    """
    
    df = pd.read_csv(file_path+ '/Lasso_Model_Input_Variables_1979_2023_v2.csv')

    # Create a mask to drop missing yields
    yield_mask = df['yield_kg_ha'].notna()

    # Apply the mask to the current dataframe
    df = df[yield_mask].copy()

    # Features and target variables
    cols_exclude = ["county", "year", "yield_kg_ha"]
    feature_cols = [c for c in df.columns if c not in cols_exclude]

    # Calculate squared terms and add to DataFrame
    #squared terms to allow non-linear relationships
    
    for col in feature_cols:
        df[f"{col}_sq"] = df[col]**2

    squared_cols = [f"{c}_sq" for c in feature_cols]
    feature_cols += squared_cols


    # County fixed effects (0ne-hot encoding)
    county_dummies = pd.get_dummies(df["county"], prefix="county", drop_first=False).astype(int)

    # Time trend within each county
    time_trend = df[["year"]].values - df["year"].min() + 1  # starts from 1
    time_trend_df = county_dummies.multiply(time_trend, axis=0)
    time_trend_df.columns = [c.replace("county", "trend") for c in time_trend_df.columns]

    # Combine into new feature DataFrame
    df_nonclim = pd.concat([county_dummies, time_trend_df], axis=1)
    df_all = pd.concat([df, df_nonclim], axis=1)

    # Update feature list
    nonclim_cols = list(df_nonclim.columns)
    feature_cols = feature_cols + nonclim_cols

    return df_all, feature_cols

# Remove influential outliers using Cook's distance

def remove_outliers_with_cooks_distance(X, Y, df_all):
    """
    Removes influential points based on Cook's distance.
    This is done before fitting any lasso model because extremly influential points can distort Lasso coefficients and cross-validation results
    """
    
    X_const = sm.add_constant(X)
    model_sm = sm.OLS(Y, X_const).fit()
    infl = model_sm.get_influence()
    cooks_d = infl.cooks_distance[0]
    threshold = 4 / len(Y)   # rule of thumb
    mask = cooks_d < threshold
    return X[mask], Y[mask], df_all.loc[mask].copy()

def build_lasso_pipeline(trial):
    """
    Constructs a pipeline:
    a. Standardize features
    b. Fit lasso with internal cross-validation (LassoCV)
    We pass the trial as random_state to LassoCV so that different HPC jobs (diff trials) produce reproducible but different model fits.
    """
    
    alphas = [5,4,3,2,1,0.5,0.1,0.05,0.01,0.005,0.001]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(
            alphas=alphas, 
            cv=5, 
            max_iter=int(1e7), 
            random_state=trial
        ))
    ])
    return pipe

def run_single_train_test(X, Y, trial):
    """
    This fits one lasso model using as single 70/30 split.
    This step 
        - gives use one reference model
        - one set of coefficients
        - on trian/test performatnce
        - This is not the final result - its just a baseline
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=trial
    )
    pipe = build_lasso_pipeline(trial)
    pipe.fit(X_train, y_train)

    #predictions

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    
    #Evaluate
    r2_train = pipe.score(X_train, y_train)
    r2_test = pipe.score(X_test, y_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return pipe, r2_train, r2_test, rmse_train, rmse_test, X_train, X_test, y_train, y_test

def run_70_30_validation(X, Y, feature_cols, trial, n_iterations=100):
    """
    This function repeats the 70/30 split 1000 times and fits Lasso each time.
    Logic of this Step
        -Instead of just trusting one random split,
         we repate the process 1000 times to assess:
            - How stable R2 is
            - How stable coefficients are
            - Whether results depend too much on a single split
        Flow inside this function:
            For each iteration i:
                a. We name a new random split 70/30 split (random_state - i_
                b. We fit a new LassoCV model
                c. Then we store:
                    - R2 train/test
                    - RMSE train/test
                    - Coefficients
                    - Intercept
                    - Scaler parameters
    """

    r2_train_list = []
    r2_test_list = []
    rmse_train_list = []
    rmse_test_list = []
    coef_list = []
    intercept_list = []
    scaler_list = []
    alphas = [5,4,3,2,1,0.5,0.1,0.05,0.01,0.005,0.001]
    
    
    for i in tqdm(range(n_iterations), desc=f"Lasso 70-30 Validation (trail {trial})"):
    # Perform 70-30 train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i) # We dont use trial here, we use i instead, so each split is different

        pipe = Pipeline([
            ("scaler", StandardScaler()), 
            ("lasso", LassoCV(
                alphas=alphas, 
                cv=5, 
                max_iter=int(1e9),
                random_state=45  #fixed seed so only the split changes
            ))
        ])

        pipe.fit(X_train, y_train)

        #predictions

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        #rmse
        rmse_train_val = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test_val = np.sqrt(mean_squared_error(y_test, y_test_pred))

        rmse_train_list.append(rmse_train_val)
        rmse_test_list.append(rmse_test_val)

        #R2
        r2_train_val = pipe.score(X_train, y_train)
        r2_test_val = pipe.score(X_test, y_test)

        r2_train_list.append(r2_train_val)
        r2_test_list.append(r2_test_val)
        
        #coefficients 
        lasso = pipe.named_steps['lasso']
        scaler = pipe.named_steps['scaler']
        
        coef_list.append(lasso.coef_)
        intercept_list.append(lasso.intercept_)
        scaler_list.append(scaler)

    #Aggregate coefficients within this trial
    coef_array = np.vstack(coef_list)
    mean_coef_std = np.mean(coef_array, axis=0)
    mean_intercept_std = np.mean(intercept_list)

    mean_scale = np.mean([s.scale_ for s in scaler_list], axis=0)
    mean_mean = np.mean([s.mean_ for s in scaler_list], axis=0)

    mean_coef_unstd = mean_coef_std / mean_scale
    mean_intercept_unstd = mean_intercept_std - np.sum((mean_mean / mean_scale) * mean_coef_std)

    validation_results = {
        "trial": trial,
        "r2_train": np.array(r2_train_list),
        "r2_test": np.array(r2_test_list),
        "rmse_train": np.array(rmse_train_list), 
        "rmse_test": np.array(rmse_test_list),
        "mean_r2_train": float(np.mean(r2_train_list)),
        "mean_r2_test": float(np.mean(r2_test_list)),
        "mean_rmse_train": float(np.mean(rmse_train_list)),
        "mean_rmse_test": float(np.mean(rmse_test_list)),
        "mean_coef_standardized": mean_coef_std,
        "mean_coef_unstandardized": mean_coef_unstd,
        "mean_intercept_unstandardized": mean_intercept_unstd,
        "feature_cols": feature_cols
    }
    return validation_results

#main workflow

if __name__ == "__main__":

    # Load data (once per trial)
    df_all, feature_cols = load_and_prepare_data()

    X = df_all[feature_cols].values
    Y = df_all["yield_kg_ha"].values

    #Remove outliers (once per trial)
    X, Y, df_filtered = remove_outliers_with_cooks_distance(X, Y, df_all)

    #Fit one baseline model
    pipe, r2_train, r2_test, *_ = run_single_train_test(X, Y, trial)

    #save this single model
    artifacts = {
        "pipeline": pipe, 
        "feature_cols_final": feature_cols, 
        "counties_train": sorted(df_filtered["county"].unique()),
        "year_min": df_all['year'].min()
    }

    joblib.dump(
        artifacts, 
        os.path.join(save_path, f"Lasso_pipeline_train_{trial}.joblib")
    )

    #Run repeated validation (100 splits)

    validation_results = run_70_30_validation(X, Y, feature_cols, trial, n_iterations=100)

    joblib.dump(
        validation_results, 
        os.path.join(save_path, f"lasso_70_30_trial_{trial}.joblib")
    )

    print(f"Finished trial {trial}")
    
    