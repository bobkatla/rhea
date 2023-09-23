import pandas as pd
import os
import glob
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import math

b_x = scipy.stats.boxcox


def process_from_census_data(normalise=True, boxcox=True):
    # This is simple to get the census data clean (assuming all shape the same, need to be quick)
    all_files =  glob.glob(os.path.join("./" , "*POA*"))
    # remove header and footer from ABS
    total_df = pd.read_csv("total.csv", skiprows=9, skipfooter=7, engine='python')
    total_df = total_df.dropna(axis=1, how='all')
    total_df.index = total_df.index.map(lambda r: r.replace(", VIC", ""))
    ls_df = [total_df]
    for f in all_files:
        df = pd.read_csv(f, skiprows=9, skipfooter=7, engine='python')
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, thresh=6)
        df = df[:-1]
        df = df.drop(columns=["Total"])
        first_row = df.columns[0]
        df[first_row] = df[first_row].apply(lambda r: r.replace(", VIC", ""))
        df = df.set_index(first_row)

        if df.index.name == "Fuel type":
            df = df[["Electric"]]

        elif df.index.name == "NPRD Number of Persons Usually Resident in Dwelling":
            df = df.drop(columns=["Not applicable"])
        
        elif df.index.name == "STRD Dwelling Structure":
            df = df.drop(columns=["Not applicable"])

            ls_town_house = [name for name in df.columns if "townhouse" in name]
            ls_flat = [name for name in df.columns if "Flat or apartment" in name]
            ls_others = [name for name in df.columns if name not in ls_town_house and name not in ls_flat and name not in ["Not stated", "Separate house"]]

            hold_df = pd.DataFrame()
            hold_df["Separate house"] = df["Separate house"]
            hold_df["Terrace/Townhouse"] = df[ls_town_house].sum(axis=1, numeric_only=True)
            hold_df["Flat or apartment"] = df[ls_flat].sum(axis=1, numeric_only=True)
            hold_df["Other"] = df[ls_others].sum(axis=1, numeric_only=True)
            hold_df["Missing"] = df["Not stated"]
            df = hold_df
      
        elif df.index.name == "HIND Total Household Income (weekly)":
            df = df.drop(columns=["Not applicable", "Negative income", "Partial income stated", "All incomes not stated"])
         
        elif df.index.name == "VEHRD Number of Motor Vehicles (ranges)":
            df = df.drop(columns=["Not applicable", "Not stated"])
        
        elif df.index.name == "TENLLD Tenure and Landlord Type":
            df = df.drop(columns=["Tenure type not applicable", "Tenure type not stated"])
            ls_rent = [name for name in df.columns if "Rented" in name]
            hold_df = pd.DataFrame()
            hold_df["Fully Owned"] = df['Owned outright']
            hold_df["Being Purchased"] = df['Owned with a mortgage']
            hold_df["Being Rented"] = df[ls_rent].sum(axis=1, numeric_only=True)
            # Cannot identify which part of occupied Rent-Free belong to
            hold_df["Something Else"] = df["Other tenure type"]
            df = hold_df
        
        df.index.name = "POA"
        ls_df.append(df)
    final_df = pd.concat(ls_df, axis=1)
    final_df = final_df.dropna(axis=0, thresh=10)

    # Normalisation
    if normalise:
        for col in final_df.columns:
            if col != "POA (EN)":
                final_df[col]= final_df[col] / final_df["POA (EN)"]
    final_df = final_df.drop(columns=["POA (EN)"])
    
    # box-cox to make it more "normal"
    if boxcox:
        for col in final_df.columns:
            final_df[col] = final_df[col].apply(lambda r: r+1)
            re_bx = b_x(final_df[col])
            print(f"{col}'s lambda val: {re_bx[1]}")
            final_df[col] = re_bx[0]

            # print(col)
            # plt.hist(final_df[col], bins=20)
            # plt.show()
    return final_df


def cal_corr(df, target, method="pearson") -> tuple:
    target_seri = df[target]
    results = []
    f = scipy.stats.pearsonr
    if method != "pearson":
        if method == "kendall":
            f = scipy.stats.kendalltau
        elif method == "spearman":
            f = scipy.stats.spearmanr
        else:
            raise ValueError
    
    for col in df.columns:
        score = f(df[col], target_seri)
        results.append((col, score))
    return results

from sklearn import preprocessing
from sklearn import utils

def cal_MI(df, target) -> tuple:
    target_seri = df[target]
    #convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(target_seri)
    results = mutual_info_classif(df, y_transformed)
    return [(col, re) for col, re in zip(df.columns, results)]


def main():
    b_x = scipy.stats.boxcox

    final_df = process_from_census_data(normalise=False, boxcox=False)
    final_df.to_csv("combine_census_HH.csv", index=False)
    # final_df = pd.read_csv("output/combine_census_HH.csv")

    method = "pearson"
    results = cal_corr(final_df, target="Electric", method=method)
    results.sort(key=lambda val: val[1][0], reverse=True)

    ordered_results = []
    for val in results:
        ordered_results.append((val[0], val[1][0], val[1][1]))
    
    corr_df = pd.DataFrame(ordered_results, columns=["Att", f"{method}_score", "p_value"])
    corr_df.index.name = "ranking"
    # corr_df.to_csv(f"output/att_rank_{method}_afterbx.csv")

    # Cal by MI
    # MI_results = cal_MI(final_df, "Electric")
    # MI_results.sort(key=lambda val: val[1], reverse=True)
    # MI_df = pd.DataFrame(MI_results, columns=["Att", "MI_score"])
    # MI_df.index.name = "ranking"
    # MI_df.to_csv("output/att_rank_MI.csv")


    # final_df.plot(x="$1-$149 ($1-$7,799)", y="$4,500-$4,999 ($234,000-$259,999)", kind="scatter")
    # plt.show()

    # syn_pop = pd.read_csv("new_syn_2021_HH.csv", index_col=0)
    # for col in syn_pop.columns:
    #     print(col)
    #     print(syn_pop[col].unique())
    

if __name__ == "__main__":
    main()