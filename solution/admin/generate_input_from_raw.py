# Script to transform raw data (with FR label) to the EN labels, also to drop columns.

# %%
import os
import s3fs
import pandas as pd

# Create filesystem object
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


# %%

def data_fr_raw_to_en(file_names_list, mapping, file_out_name):
    BUCKET = "projet-funathon"
    list_df = []
    cols_to_keep = list(mapping_fr_en.keys())

    for file_name in file_names_list:
        FILE_PATH_S3 = BUCKET + "/" + "2026/project1/data/0_raw/" + file_name

        with fs.open(FILE_PATH_S3, mode="rb") as file_in:
            df_raw = pd.read_parquet(file_in)

        list_df.append(df_raw[cols_to_keep])

    df_en = pd.concat(list_df)
    df_en = df_en.rename(columns=mapping)

    FILE_PATH_OUT_S3 = "s3://" + BUCKET + "/2026/project1/data/1_input/" + file_out_name

    df_en.to_parquet(FILE_PATH_OUT_S3, index=False)

    return df_en


# %%
mapping_fr_en = {
  "datemut": "trans_date",
  "anneemut": "trans_year",
  "moismut": "trans_month",
  "valeurfonc": "price",
  "dteloc": "prop_type",
  "jannath": "prop_year_harm",
  "ccodep": "prop_loc_dep",
  "depcom": "prop_loc_citycode",
  "x": "prop_loc_x",
  "y": "prop_loc_y",
  "distance_ltm": "dist_tosea",
  "dnbniv": "n_floors",
  "dnbbai": "n_bath",
  "dnbdou": "n_show",
  "dnblav": "n_sink",
  "dnbwc": "n_wc",
  "dnbppr": "n_mrooms",
  "dnbsam": "n_eatr",
  "dnbcha": "n_slr",
  "dnbcu8": "n_kit8",
  "dnbcu9": "n_kit9",
  "dnbsea": "n_washr",
  "dnbann": "n_ancrooms",
  "dnbpdc": "n_rooms",
  "dsupdc": "farea",
  "geaulc": "has_water",
  "gelelc": "has_elec",
  "gesclc": "stair",
  "ggazlc": "has_gas",
  "gasclc": "has_elevator",
  "gchclc": "has_cheating",
  "gvorlc": "has_rchute",
  "gteglc": "has_mdrainage",
  "dniv": "nth_floor",
  "dcntsol": "s_land_artif",
  "dcntagri": "s_land_agri",
  "dcntnat": "s_land_nat",
  "nb_garages": "n_garage",
  "nb_piscines": "n_pool",
  "nb_terrasses": "n_terrace",
  "nb_greniers": "n_attic",
  "nb_caves": "n_basmt",
  "nb_autresdep": "n_otherannex"}

data_fr_raw_to_en(
    ["transactions_houses.parquet", "transactions_flats.parquet"],
    mapping_fr_en,
    "transactions_EN.parquet")

# %%
