import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

from utils import process_from_census_data

import scipy.stats
b_x = scipy.stats.boxcox


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