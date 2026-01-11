import pandas as pd, numpy as np
from typing import Dict, Tuple, Literal
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from missforest import MissForest

def clean_astro_data(
    train_data: pd.DataFrame,
    # test_data: pd.DataFrame,
    impute_mode: Literal["mean", "rf"] = "rf",
    *,
    rf_estimators: int = 200,
    rf_random_state: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Impute missing values (mean or RF-based) and standardize features
    using parameters fit on the training data only.

    Parameters
    ----------
    train_data, test_data : pd.DataFrame
        DataFrames with identical columns (numeric features expected).
    impute_mode : {"mean", "rf"}
        - "mean": per-column mean imputation
        - "rf"  : Random-Forest-based IterativeImputer (missForest-like)
    rf_estimators : int
        Number of trees for the RF imputer (when impute_mode == "rf").
    rf_random_state : int
        Random seed for the RF imputer (when impute_mode == "rf").

    Returns
    -------
    dict with:
        'train' : imputed & standardized training DataFrame
        'test'  : imputed & standardized test DataFrame
    """
    # --- sanity checks
    # if list(train_data.columns) != list(test_data.columns):
    #     raise ValueError("train_data and test_data must have identical columns and order.")

    cols = train_data.columns
    # Work on copies to avoid mutating caller's data
    Xtr = train_data.copy()
    # Xte = test_data.copy()

    # --- Imputation
    if impute_mode == "mean":
        imp = SimpleImputer(strategy="mean")
        Xtr_imp = pd.DataFrame(imp.fit_transform(Xtr), columns=cols, index=Xtr.index)
        # Xte_imp = pd.DataFrame(imp.transform(Xte), columns=cols, index=Xte.index)

    elif impute_mode == "rf":
        # IterativeImputer with RF (missForest-like). Fit on TRAIN ONLY, then transform both.
        mf = MissForest()
        
        Xtr_imp = pd.DataFrame(mf.fit_transform(Xtr), columns=cols, index=Xtr.index)
        # Xte_imp = pd.DataFrame(mf.transform(Xte), columns=cols, index=Xte.index)

    else:
        raise ValueError("impute_mode must be 'mean' or 'rf'.")

    # --- Standardize (center/scale) using TRAIN params only
    # scaler = StandardScaler(with_mean=True, with_std=True)
    # Xtr_std = pd.DataFrame(scaler.fit_transform(Xtr_imp), columns=cols, index=Xtr_imp.index)
    # Xte_std = pd.DataFrame(scaler.transform(Xte_imp), columns=cols, index=Xte_imp.index)

    # return {"train": Xtr_imp, "test": Xte_std}
    return Xtr_imp

astro_df = pd.read_csv('/user/bnc2119/drd/astro_clean_data.csv')
cleaned_astro_df = clean_astro_data(astro_df)
print(cleaned_astro_df.head())
cleaned_astro_df.to_csv('/user/bnc2119/drd/astro_data_final.csv', index=False)
