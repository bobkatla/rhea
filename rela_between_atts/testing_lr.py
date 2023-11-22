from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, BayesianRidge, Perceptron, ARDRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from feature_importance import process_from_census_data
import pandas as pd
import numpy as np


def holding_land():
    NotImplemented
    # hold = pd.DataFrame(X.columns, columns=["Census_name"])
    # hold["Sample_att"] = "Namecheck"
    # hold["Sample_val"] = "Nananan"

    # hold['Sample_att'].mask(hold['Census_name'].str.contains("person"), "hhszie", inplace=True)
    # hold['Sample_att'].mask(hold['Census_name'].str.contains("vehicle"), "totalvehs", inplace=True)
    # hold['Sample_att'].mask(hold['Census_name'].str.contains("\$"), "hhinc", inplace=True)
    # print(hold)
    # hold.to_csv("data/dict_cross.csv", index=False)


def condition_d_cross(val, sample_val):
    condition = None
    if "8+" == sample_val:
        # This is a plus, only special case for hhsize
        thres_v = sample_val.replace("+", "")
        thres_v = float(thres_v)
        condition = float(val) >= thres_v
    else:
        # Equal check
        condition = str(val) == str(sample_val)

    assert condition is not None
    return condition


def convert_sample_data_to_census_format(disag_data, dict_cross):
    # Dict_cross will have 3 columns: Census_name, Sample_att and Sample_val
    # Change the dict_cross for changing your assessment
    ls_atts_need_to_assess = dict_cross["Sample_att"].unique()
    d_data = disag_data[ls_atts_need_to_assess]
    d_data["hhid"] = d_data.index
    ls_name_census = list(dict_cross["Census_name"].unique())
    n_census_col = len(ls_name_census)
    # Create new df and return based on np array
    result_arr = []
    for index, row in d_data.iterrows():
        print(f"Doing {index}")
        val_convered_census = [0 for _ in range(n_census_col)]
        for att in ls_atts_need_to_assess:
            census_name_check = None
            val = row[att]
            check = dict_cross[dict_cross["Sample_att"]==att]
            for census_name, sample_val in zip(check["Census_name"], check["Sample_val"]):
                condition = condition_d_cross(val, sample_val)
                if condition:
                    census_name_check = census_name
            if census_name_check is None:
                print(att, val)
            assert census_name_check is not None
            idx_census = ls_name_census.index(census_name_check)
            val_convered_census[idx_census] = 1
        val_convered_census.append(row["hhid"])
        result_arr.append(val_convered_census)

    ls_name_census.append("hhid")
    result_converted_df = pd.DataFrame(np.array(result_arr), columns=ls_name_census)
    return d_data, result_converted_df
    

def get_model_ev_pred(method_of_lr, X, y):
    model = None
    if method_of_lr == 'lr':
        model = LinearRegression()
    elif method_of_lr == 'logis':
        model = LogisticRegression(random_state=0)
    elif method_of_lr == 'SDG':
        model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    elif method_of_lr == 'baye':
        model = BayesianRidge()
    elif method_of_lr == "forest":
        model = RandomForestRegressor(max_depth=6, random_state=0)
    elif method_of_lr == "GraBoost":
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,loss='squared_error')
    
    model.fit(X, y)
    return model


def get_ev_pred(model, to_pred_data):
    if "hhid" in to_pred_data.columns:
        to_pred_data = to_pred_data.drop(columns=["hhid"])
    return model.predict(to_pred_data)


def combine_and_test_diff_methods():
    # Get the to predict data
    d_cross = pd.read_csv("data/dict_cross.csv")
    h_sample_test = pd.read_csv("data/syn_hh_final.csv")
    # h_sample_test = pd.read_csv(r".\data\H_sample.csv")
    h_sample_test = h_sample_test[[
        "totalvehs",
        "hhsize",
        "dwelltype",
        "owndwell",
        "hhinc",
    ]]
    print("Converting the population to the needed format")
    process_sample_data, converted_sample_data = convert_sample_data_to_census_format(h_sample_test, d_cross)
    process_sample_data.to_csv("./output/POA_to_pred.csv", index=False)
    converted_sample_data.to_csv("./output/POA_converted_to_input.csv", index=False)
    # process_sample_data = pd.read_csv("output/EV_pred/POA_to_pred.csv")
    # converted_sample_data = pd.read_csv("output/EV_pred/POA_converted_to_input.csv")

    df = process_from_census_data(geo_lev="POA", boxcox=False)
    X = df.drop(columns="Electric").astype("float")
    y = df["Electric"].astype("float")

    final_re = {}
    for method_of_lr in ["forest", 'baye', 'lr', 'GraBoost']:
        print(f"DOING fit model {method_of_lr}")
        model = get_model_ev_pred(method_of_lr, X, y)
        print("Predicting")
        ev_re = get_ev_pred(model, converted_sample_data)
        print("OUTPUT")
        final_re[method_of_lr] = ev_re
    
    for re in final_re:
        process_sample_data[f"EV_pred_{re}"] = final_re[re]

    # Forgot to add the location zone
    process_sample_data.to_csv("./output/POA_EV_pred_all.csv", index=False)


def process_pred_pearson():
    geo_lev = "POA"
    ori_df = pd.read_csv("data/synthetic_2021_HH_POA.csv", index_col=0)
    
    ori_df = ori_df.reset_index()
    to_pred = pd.read_csv(f"output/EV_pred/{geo_lev}_to_pred.csv")
    ls_cols = to_pred.columns.drop("hhid")
    to_pred[geo_lev] = ori_df[geo_lev]

    pearson_val = pd.read_csv(f"output/feature_importance/att_rank_pearson.csv")
    d_cross = pd.read_csv("data/dict_cross.csv")

    dict_pear_val = dict(zip(pearson_val["Att"], pearson_val["pearson_score"]))
    dict_val_final = {}
    for att in ls_cols:
        hold_df = d_cross[d_cross["Sample_att"]==att]
        for val in to_pred[att].unique():
            for census_name, state in zip(hold_df["Census_name"], hold_df["Sample_val"]):
                condition = condition_d_cross(val, state)
                if condition:
                    dict_val_final[(att, str(val))] = dict_pear_val[census_name]
                    break

    def f_row(row):
        final_score = 0
        for col in ls_cols:
            cell_v = str(float(row[col])) if str(row[col]).replace(".", "").isnumeric() else row[col]
            val = dict_val_final[(col, cell_v)]
            final_score += float(val)
        return final_score
    
    to_pred["EV_pred_Pearsons"] = to_pred.apply(f_row, axis=1)

    print(to_pred)
    to_pred.to_csv(f"{geo_lev}_Pearson_pred.csv", index=False)


def quick_process(geo_lev = "POA"):
    file_loc = "output/EV_pred"
    df_all = pd.read_csv(f"{file_loc}/{geo_lev}_EV_pred_all.csv")
    df_pearson = pd.read_csv(f"{file_loc}/{geo_lev}_Pearson_pred.csv")
    df_all[geo_lev] = df_pearson[geo_lev]
    df_all["EV_pred_Pearsons"] = df_pearson["EV_pred_Pearsons"]
    df_all.to_csv(f"{file_loc}/{geo_lev}_EV_pred_all.csv", index=False)


def main():
    # process_pred_pearson()
    # quick_process("SA1")
    combine_and_test_diff_methods()
    # for method_of_lr in ["GraBoost"]:
    #     print(f"RUNNING {method_of_lr}")
    #     run_lr_EV(method_of_lr)
    #     new_pred = pd.read_csv(f"EV_pred_ori_{method_of_lr}.csv")
    #     new_pred = new_pred.sort_values(by=["EV_pred"])
    #     print(f"WORST for {method_of_lr}")
    #     print(new_pred.head(20))
    #     print(f"BEST for {method_of_lr}")
    #     print(new_pred.tail(20))


if __name__ == "__main__":
    main()