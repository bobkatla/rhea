from sklearn.linear_model import LinearRegression
from main import process_from_census_data
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
        val_convered_census = [0 for _ in range(n_census_col)]
        for att in ls_atts_need_to_assess:
            census_name_check = None
            val = row[att]
            check = dict_cross[dict_cross["Sample_att"]==att]
            for census_name, sample_val in zip(check["Census_name"], check["Sample_val"]):
                condition = None
                if "-" in sample_val and "Rent" not in sample_val:
                    # This is a range
                    min_v, max_v = sample_val.split("-")
                    min_v, max_v = float(min_v), float(max_v)
                    condition = float(val) >= min_v and float(val) <= max_v
                elif "+" in sample_val:
                    # This is a plus
                    thres_v = sample_val.replace("+", "")
                    thres_v = float(thres_v)
                    condition = float(val) >= thres_v
                else:
                    # Equal check
                    if sample_val.isdigit():
                        val = float(val)
                        sample_val = float(sample_val)
                    condition = str(val) == str(sample_val)

                assert condition is not None
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


def run_lr_EV():
    d_cross = pd.read_csv("data/dict_cross.csv")
    h_sample_test = pd.read_csv("data/new_syn_2021_HH.csv", index_col=0)
    process_sample_data, converted_sample_data = convert_sample_data_to_census_format(h_sample_test, d_cross)
    
    df = process_from_census_data(boxcox=False)
    X = df.drop(columns="Electric")
    y = df["Electric"]

    model = LinearRegression()
    model.fit(X, y)

    check_X = converted_sample_data.drop(columns=["hhid"])
    re_pred = model.predict(check_X)
    process_sample_data["EV_pred"] = re_pred
    process_sample_data["SA1"] = h_sample_test["SA1"]

    process_sample_data.to_csv("results_EV_pred.csv", index=False)


def main():
    run_lr_EV()
    # new_pred = pd.read_csv("results_EV_pred.csv")
    # new_pred = new_pred.sort_values(by=["EV_pred"])
    # print(new_pred.tail(40))


if __name__ == "__main__":
    main()