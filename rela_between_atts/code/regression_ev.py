from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import statistics as stat
from pathlib import Path



def get_model_ev_pred(method_of_lr, X, y, action="fit"):
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
    
    if action == "fit":
        model.fit(X, y)
        return model, None
    else:
        score = cross_val_score(model, X, y, cv=10, scoring=action)
        return model, score
    

def get_ev_pred(model, to_pred_data):
    return model.predict(to_pred_data)

data_folder = Path("rela_between_atts/data/new_data")

def get_processed_data(type="hh"):    
    data_dir = Path(data_folder / "processed")
    if type == "hh":
        census_file = data_dir / "processed_census_hh.csv"
        to_predict = data_dir / "matrix_hh.csv" # maybe keep the first col as id
    elif type == "pp":
        census_file = data_dir / "processed_census_pp.csv"
        to_predict = data_dir / "matrix_pp.csv" # maybe keep the first col as id
    census_input = pd.read_csv(census_file, index_col=0)
    predict_syn = pd.read_csv(to_predict, index_col=0)

    # NOTE: filter based on this
    if type =="pp":
        predict_syn = predict_syn[predict_syn["nolicence_Some Licence"]==1]
        drop_cols = [x for x in predict_syn.columns if "nolicence_" in x or "relationship_" in x]
        predict_syn = predict_syn.drop(columns=drop_cols)
    elif type == "hh":
        predict_syn = predict_syn[predict_syn["totalvehs_0"]==0]
        predict_syn = predict_syn.drop(columns=["totalvehs_0"])
        census_input = census_input.drop(columns=["totalvehs_0"])
    return census_input, predict_syn


def main():
    type_check = "pp"
    targer_col = "Vehicle_Electric_ratio"
    census_input, predict_syn = get_processed_data(type_check)
    X = census_input.drop(columns=targer_col).astype("float")
    y = census_input[targer_col].astype("float")

    # Sort to be same order
    X = X.reindex(sorted(X.columns), axis=1)
    predict_syn = predict_syn.reindex(sorted(predict_syn.columns), axis=1)
    assert list(X.columns) == list(predict_syn.columns)
    final_re = {}
    for method_of_lr in ["forest", 'baye', 'lr', 'GraBoost']:
    # print(f"DOING scoring model {method_of_lr}")
    # final_score = []
    # for score in ls_metrics:
    #     _, scores = get_model_ev_pred(method_of_lr, X, y, action=score)
    #     final_score.append(stat.mean(scores))
    # final_scoring_methods[method_of_lr] = final_score

        print(f"DOING fit model {method_of_lr}")
        model, _ = get_model_ev_pred(method_of_lr, X, y, action="fit")
        print("Predicting")
        ev_re = get_ev_pred(model, predict_syn)
        print("OUTPUT")
        final_re[method_of_lr] = ev_re
    syn_dir = Path(data_folder / "syn_pop")
    if type_check == "hh":
        ori_syn = pd.read_csv(syn_dir / "syn_hh_ipu.csv")
    else:
        ori_syn = pd.read_csv(syn_dir / "syn_pp_ipu.csv")
    
    # Need to update this
    for method, val in final_re.items():
        ori_syn[f"EV_score_{method}"] = -100000000
        ori_syn.loc[predict_syn.index, f"EV_score_{method}"] = val

    ori_syn.to_csv(data_folder / "final" / f"syn_{type_check}_with_ev_score.csv")


if __name__ == "__main__":
    main()