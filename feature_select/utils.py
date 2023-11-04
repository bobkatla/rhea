import numpy as np
import pandas as pd
import os
import glob


def TRS(initial_weights, know_tot=None):
    # Desired total constraint
    desired_total = know_tot if know_tot else sum(initial_weights)

    # Step 1: Truncate the weights to integers
    truncated_weights = np.floor(initial_weights).astype(int)

    # Step 2: Calculate the discrepancy
    total_truncated = np.sum(truncated_weights)
    discrepancy = desired_total - total_truncated

    # Step 3: Replicate individuals to match the constraint
    if discrepancy > 0:
        # Calculate fractional parts
        fractional_parts = initial_weights - truncated_weights
        
        # Replicate individuals in proportion to their fractional parts
        replication_probs = fractional_parts / np.sum(fractional_parts)
        num_replications = np.random.multinomial(int(discrepancy), replication_probs)
        truncated_weights += num_replications

    # Step 4: Sample individuals if there is an excess
    if discrepancy < 0:
        excess_indices = np.where(truncated_weights > 0)[0]
        excess_weights = truncated_weights[excess_indices]
        
        # Calculate sampling probabilities based on truncated weights
        sampling_probs = excess_weights / np.sum(excess_weights)
        
        # Randomly sample individuals to reduce excess
        num_samples = np.random.multinomial(abs(discrepancy), sampling_probs)
        truncated_weights[excess_indices] -= num_samples

    # Step 5: Your final truncated and rounded integer weights
    return truncated_weights



def process_from_census_data(geo_lev='POA', normalise=True, boxcox=True, keep_same=False, return_tot=False):
    # This is simple to get the census data clean (assuming all shape the same, need to be quick)
    all_files =  glob.glob(os.path.join("./data" , f"{geo_lev}*"))
    # remove header and footer from ABS
    total_df = pd.read_csv(f"data/total_{geo_lev}.csv", skiprows=9, skipfooter=7, engine='python')
    total_df = total_df.dropna(axis=1, how='all')
    total_df.index = total_df.index.map(lambda r: r.replace(", VIC", ""))
    ls_df = [total_df]
    for f in all_files:
        df = pd.read_csv(f, skiprows=9, skipfooter=7, engine='python')
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, thresh=6)
        df = df[:-1]
        if "Total" in df.columns:
            df = df.drop(columns=["Total"])
        first_row = df.columns[0]
        df[first_row] = df[first_row].apply(lambda r: r.replace(", VIC", ""))
        df = df.set_index(first_row)

        if keep_same:
            df =df.add_prefix(f"{df.index.name}__")
        else:
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
        
        df.index.name = geo_lev
        ls_df.append(df)
    final_df = pd.concat(ls_df, axis=1)
    final_df = final_df.dropna(axis=0, thresh=10)

    # Normalisation
    tot_df = final_df[f"{geo_lev} (EN)"]
    if normalise:
        for col in final_df.columns:
            if col != f"{geo_lev} (EN)":
                final_df[col]= final_df[col].astype(float) / final_df[f"{geo_lev} (EN)"].astype(float)
    final_df = final_df.drop(columns=[f"{geo_lev} (EN)"])
    
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

    return (final_df, tot_df) if return_tot else final_df

