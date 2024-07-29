import pandas as pd
import numpy as np
from pathlib import Path


data_folder = Path("rela_between_atts/data/new_data")


# Note that our data now has clear value
def import_processed_census_data():
    file_census_hh = Path(data_folder / "census/hh_marginals_percen.csv")
    file_census_pp = Path(data_folder / "census/person_marginals_percen.csv")

    census_hh = pd.read_csv(file_census_hh, header=[0,1])
    census_pp = pd.read_csv(file_census_pp, header=[0,1])

    drop_cols_hh = [x for x in census_hh.columns if "zone_id" in x[0] or "geog" in x[0]]
    drop_cols_pp = [x for x in census_pp.columns if "zone_id" in x[0] or "geog" in x[0]]

    census_hh = census_hh.drop(columns=drop_cols_hh)
    census_pp = census_pp.drop(columns=drop_cols_pp)

    census_hh.columns = [f"{x[0]}_{x[1]}" for x in list(zip(census_hh.columns.get_level_values(0), census_hh.columns.get_level_values(1)))]
    census_pp.columns = [f"{x[0]}_{x[1]}" for x in list(zip(census_pp.columns.get_level_values(0), census_pp.columns.get_level_values(1)))]
    return census_hh, census_pp


def import_syn_data():
    file_syn_hh = Path(data_folder / "syn_pop/syn_hh_ipu.csv")
    file_syn_pp = Path(data_folder / "syn_pop/syn_pp_ipu.csv")

    syn_hh = pd.read_csv(file_syn_hh)
    syn_pp = pd.read_csv(file_syn_pp)
    syn_hh = syn_hh.drop(columns=["serialno", "sample_geog", "cat_id", "geog"]).reset_index(drop=True)
    syn_pp = syn_pp.drop(columns=["serialno", "sample_geog", "cat_id", "geog", "household_id"]).reset_index(drop=True)
    syn_hh["id"] = syn_hh.index
    syn_pp["id"] = syn_pp.index
    return syn_hh, syn_pp


def convert_sample_data_to_census_format(disag_data: pd.DataFrame):
    """The idea of this will turn each record into the matrix format of census, where mostly is zero except some is 0"""
    # Simple idea, just fill a columns
    assert "id" in disag_data.columns # use to determine value
    assert not disag_data["id"].duplicated().any()
    disag_data = disag_data.set_index("id")
    cols_expected = []
    for col in disag_data.columns:
        for val in disag_data[col].unique():
            cols_expected.append(f"{col}_{val}")
    result_matrix = pd.DataFrame(np.zeros((len(disag_data), len(cols_expected))), columns=cols_expected, index=disag_data.index)
    # for now just run all, can improve speed later by choosing only specific one
    # convert each cell into att_val (match with census)
    def convert_to_census_name(col):
        return f"{col.name}_" + col.astype(str)
    converted_disag = disag_data.apply(convert_to_census_name, axis=0)
    
    def update_result_matrix(r):
        result_matrix.loc[r.name, r.values] = 1
    converted_disag.apply(update_result_matrix, axis=1)
    return result_matrix


def main():
    census_hh, census_pp = import_processed_census_data()
    syn_hh, syn_pp = import_syn_data()
    matrix_hh = convert_sample_data_to_census_format(syn_hh)
    matrix_pp = convert_sample_data_to_census_format(syn_pp)
    matrix_hh.to_csv(data_folder / "processed/matrix_hh.csv")
    matrix_pp.to_csv(data_folder / "processed/matrix_pp.csv")
    census_hh.to_csv(data_folder / "processed/processed_census_hh.csv")
    census_pp.to_csv(data_folder / "processed/processed_census_pp.csv")


if __name__ == "__main__":
    main()
    
