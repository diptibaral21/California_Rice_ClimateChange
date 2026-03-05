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

file_path = "/group/moniergrp/dbaral/run_project/input_data/gridmet_hist_model_input"
save_path = "/group/moniergrp/dbaral/run_project/output_data/historical_model"

#helper functions

def load_and_prepare_data():
    """
    This function loads the raw dataset and constructs all features, including:
        - squared terms for non-linear effects
        - county fixed effects 
        - time trends per county
    This step is done once per job_id
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

    # County fixed effects
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


# def fit_and_save_full_sample_lasso(X, Y, feature_cols, save_path, model_name, trial=45):
#     """
#     Fit full-sample Lasso for a given dataset (trial).
#     - trial is used as random_state in LassoCV for reproducibility.

#     Here we fit the full-sample Lasso model and save the following:
#         - full pipelne(joblib)
#         - standardized lasso coefficients (CSV)
#         - unstandardized lasso coefficients (CSV)
#         - model metadata (alpha, intercept, R2)
#     """
#     alphas = [5,4,3,2,1,0.5,0.1,0.05,0.01,0.005,0.001]
#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("lasso", LassoCV(
#             alphas=alphas,
#             cv=5,
#             max_iter=int(1e6),
#             random_state=45  
#         ))
#     ])
#     #fit the model
#     pipe.fit(X, Y)

#     #extract components
#     lasso = pipe.named_steps['lasso']
#     scaler = pipe.named_steps['scaler']

#     #coefficients
#     coef_std = lasso.coef_
#     coef_unstd = coef_std / scaler.scale_
#     intercept_unstd = lasso.intercept_ - np.sum((scaler.mean_ / scaler.scale_) * coef_std)

#     #save full pipeline 
#     os.makedirs(save_path, exist_ok=True)
#     joblib.dump(pipe, os.path.join(save_path, f"{model_name}_pipeline.pkl"))

#     #save standardized coefficients

#     coef_std_df = pd.Series(coef_std, index=feature_cols)
#     coef_std_df.to_csv(os.path.join(save_path, f"{model_name}_standardized_coef.csv"), header = ["coefficient"])

#     #save unstandardized coefficients

#     coef_unstd_df = pd.Series(coef_unstd, index=feature_cols)
#     coef_unstd_df.to_csv(os.path.join(save_path, f"{model_name}_unstandardized_coef.csv"), header = ["coefficient"])

#     #save metadata

#     metadata = {
#         "gcm": gcm, 
#         "scenario": scenario,
#         "alpha_selected": lasso.alpha_,
#         "intercept_unstandardized": intercept_unstd,
#         "r_squared": pipe.score(X,Y)
#     }

        
def run_70_30_validation(X, Y, df_filtered, feature_cols, trial, n_iterations=1000):
    """
    This function repeats the 70/30 split 1000 times and fits Lasso each time.
    70/30 split is done within each country because if we do 70/30 split in full dataset then there is
    chances of having a whole country missing from the training dataset
    
    Logic of this Step
        -Instead of just trusting one random split,
         we repate the process 1000 times to assess:
            - How stable R2 is
            - How stable coefficients are
            - Whether results depend too much on a single split
        Flow inside this function:
            For each iteration i:
                a. We name a new random split 70/30 split (random_state - i)
                b. We fit a new LassoCV model
                c. Then we save:
                    - Unstandardized coefficients
                    - Intercept
                    - alpha
                    - R2 train/test
                    - RMSE train/test
        Saves:
            1. lasso_1000_model_coefficents.csv
            2. lasso_1000_validation_metrics.csv
    """
    all_coef_list = []
    metrics_list = []
    
    for i in tqdm(range(1, n_iterations+1), desc=f"Running Lasso 1000 models"):
    # Perform county-wise 70-30 train/test split
        unique_counties = df_filtered['county'].unique()

        train_indices = []
        test_indices = []

        for county in unique_counties:
            county_mask = df_filtered['county'] == county
            county_indices = df_filtered[county_mask].index

            train_index_c, test_index_c = train_test_split(county_indices, test_size=0.3, random_state = i )
            train_indices.extend(train_index_c)
            test_indices.extend(test_index_c)
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = Y.loc[train_indices]
        y_test = Y.loc[test_indices]

        #fit LassoCV
        alphas = [5,4,3,2,1,0.5,0.1,0.05,0.01,0.005,0.001]
        pipe_lasso_70_30_split = Pipeline([
            ("scaler", StandardScaler()), 
            ("lasso", LassoCV(
                alphas=alphas, 
                cv=5, 
                max_iter=int(1e6),
                random_state=45  #fixed seed so only the split changes
            ))
        ])
        #fit model

        pipe_lasso_70_30_split.fit(X_train, y_train)

        lasso = pipe_lasso_70_30_split.named_steps['lasso']
        scaler = pipe_lasso_70_30_split.named_steps['scaler']

        #convert to unstandardized coefficients
        coef_std = lasso.coef_
        coef_unstd = coef_std/scaler.scale_

        intercept_unstd = (lasso.intercept_ - np.sum((scaler.mean_/scaler.scale_) * coef_std))

        #save coefficients (long format)
        coef_df_iteration = pd.DataFrame({
            'feature': feature_cols, 
            'coefficient': coef_unstd, 
            'iteration': i
        })
        
        all_coef_list.append(coef_df_iteration)
        

        #predictions and validation metrics

        y_train_pred = pipe_lasso_70_30_split.predict(X_train)
        y_test_pred = pipe_lasso_70_30_split.predict(X_test)

        #rmse
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        #R2
        r2_train = pipe_lasso_70_30_split.score(X_train, y_train)
        r2_test = pipe_lasso_70_30_split.score(X_test, y_test)


        #save csv with R2 and RMSE for each iteration

        metrics_list.append({
            'iteration': i, 
            'alpha_selected': lasso.alpha_,
            'R2_train': r2_train,
            'R2_test': r2_test,
            'RMSE_train': rmse_train,
            'RMSE_test': rmse_test
        })
    #After the loop save everything

    final_coef_df = pd.concat(all_coef_list, ignore_index = True)
    metrics_df = pd.DataFrame(metrics_list)

    final_coef_df.to_csv(os.path.join(save_path, 'lasso_1000_models_coefficients.csv'), index=False)

    metrics_df.to_csv(
        os.path.join(save_path, "lasso_1000_model_validation_metrics.csv"), index=False)
    return final_coef_df, metrics_df 
    
            
#main workflow

if __name__ == "__main__":

    # Load data (once per trial)
    df_all, feature_cols = load_and_prepare_data()

    X = df_all[feature_cols]
    Y = df_all["yield_kg_ha"]

    #Remove outliers (once per trial)
    X, Y, df_filtered = remove_outliers_with_cooks_distance(X, Y, df_all)

    # #Fit one full-sample lasso model
    # trial = 45
    # fit_and_save_full_sample_lasso(X, Y, feature_cols, save_path, model_name, trial=45)

    #Run repeated validation (1000 splits)

    validation_results = run_70_30_validation(X, Y, df_filtered, feature_cols, trial, n_iterations=1000)

    print("Finished running the model")
    
    