
# ============================================
# STEP 1 — Data exploration
#
# ============================================

# YOUR CODE HERE
# %%
import polars as pl
import plotnine as p9
from datetime import datetime, timedelta
import seaborn as sns

pl.Config.set_tbl_cols(-1)  # show all columns
pl.Config.set_tbl_rows(50)  # show 20 rows
# %%
data = pl.read_parquet('s3://confpns/synthetic-transactions/rawdata/transactions/transactions_flats_final.parquet')

# %%

data_h = pl.read_parquet("s3://confpns/synthetic-transactions/rawdata/transactions/transactions_houses_final.parquet")

# %%
if data.columns == data_h.columns:
    data_all = pl.concat([data, data_h])
# %%
#| label: paris_transactions_all_plot
(
    p9.ggplot(
        data.select(["prop_loc_dep", "x", "y", "price"]).with_columns(
                price_log=pl.col("price").log(base=10)
            ).filter(pl.col("prop_loc_dep")=="75"),
        p9.aes("x","y", colour="price_log")
    ) +
    p9.geom_point(size=0.05)+
    p9.theme_matplotlib() +
    p9.ggtitle("Localization of transactions (flat+houses) in Paris with price")
)

# %%
#| label: all_transactions_plot
(
    p9.ggplot(
        data_all.select(["x", "y"]),
        p9.aes("x","y")
    ) +
    p9.geom_point(size=0.01)+
    p9.theme_matplotlib() +
    p9.ggtitle("Localization of transactions (flat+houses) in France since 2010")
)
# %%
#| label: all_transactions_plot_hexag
(
    p9.ggplot(
        data_all.select(["x", "y"])
        .filter(
            pl.col("x") >= -5,
            pl.col("x") <= 20,
            pl.col("y") >= 30,
            pl.col("y") <= 60
        ),
        p9.aes("x","y")
    ) +
    p9.geom_point(size=0.01)+
    p9.theme_matplotlib() +
    p9.ggtitle("Localization of transactions (flat+houses) in France hexag since 2010")
)


# %%
# Retrouver la mutation 
# data_h.filter(
#             pl.col("trans_id")=="DVF+_6242255"
#             ).glimpse()


# %%
def analyse_colonnes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Retourne un DataFrame avec pour chaque colonne :
    - Nom, type, statistiques de valeurs (nulles, NaN, manquantes, valides)
    - Médiane (numérique), mode (string), date moyenne (date)
    - Min, max
    """
    resultats = []

    for col in df.columns:
        serie = df[col]
        n_total = len(serie)
        n_null = serie.null_count()

        # Calcul des NaN (spécifique aux float)
        n_nan = 0
        if serie.dtype in (pl.Float32, pl.Float64):
            n_nan = serie.is_nan().sum()
        n_valid = n_total - n_null - n_nan

        # Calcul de la médiane, mode, min, max, ou date moyenne
        if serie.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
            median = f'{serie.median():.2f}'
            val_min = f'{serie.min():.2f}'
            val_max = f'{serie.max():.2f}'
            val_001 = f'{serie.quantile(0.001):.2f}'
            val_01 = f'{serie.quantile(0.01):.2f}'
            val_05 = f'{serie.quantile(0.05):.2f}'
            val_25 = f'{serie.quantile(0.25):.2f}'
            val_75 = f'{serie.quantile(0.75):.2f}'
            val_95 = f'{serie.quantile(0.95):.2f}'
            val_99 = f'{serie.quantile(0.99):.2f}'
            val_999 = f'{serie.quantile(0.999):.2f}'
        elif serie.dtype == pl.Date:
            median = serie.median()
            val_min = serie.min()
            val_max = serie.max()
            val_001 = "."
            val_01 = "."
            val_05 = "."
            val_25 = "."
            val_75 = "."
            val_95 = "."
            val_99 = "."
            val_999 = "."
            # Calcul de la date moyenne
            dates = serie.drop_nulls().to_list()
            if dates:
                avg_date = sum((d - datetime.min.date()).days for d in dates) / len(dates)
                median = datetime.min.date() + timedelta(days=avg_date)
            else:
                median = "."
        else:  # Strings, booléens, etc.
            median = serie.mode().first()
            val_min = serie.min()
            val_max = serie.max()
            val_001 = "."
            val_01 = "."
            val_05 = "."
            val_25 = "."
            val_75 = "."
            val_95 = "."
            val_99 = "."
            val_999 = "."

        resultats.append({
            "colonne": col,
            "type": str(serie.dtype),
            "total": n_total,
            "nulls": n_null,
            "NaN": n_nan,
            "valid": n_valid,
            "min": str(val_min),
            "0.1%": val_001, 
            "1%": val_01, 
            "5%": val_05, 
            "25%": val_25, 
            "median/mode": str(median),
            "75%": val_75, 
            "95%": val_95, 
            "99%": val_99, 
            "99.9%": val_999, 
            "max": str(val_max)
        })

    return pl.DataFrame(resultats)


#%%
descr_df = analyse_colonnes(data_all)
#%%
print(descr_df)


# %%
# print(descr_df.to_pandas().to_markdown(index=False))

# %%
#| label: stat_des_plot
# Reshape the data for plotting
descr_df_long = (
    descr_df
    .filter(
        pl.col("type") != "String",
        pl.col("type") != "Date", 
        pl.col("max").str.len_chars() <= 5  # keeping only var with max <100
    )
    .unpivot(
    index='colonne',
    on=['min', '1%','5%', '25%', 'median/mode', '75%', '95%', '99%', 'max'],
    variable_name='statistic',
    value_name='value'
    )
    .cast({"value":pl.Float32})
    .filter(
        pl.col("colonne") != "x",
        pl.col("colonne") != "y",
        pl.col("colonne") != "trans_type_code",
        pl.col("colonne") != "trans_month"
    )
)
# %%
#| label: stat_des_plot
# Create the plot
(
    p9.ggplot(p9.aes(colour="statistic")) +
    p9.geom_point(
        descr_df_long.filter(pl.col('statistic') == 'min'),
        p9.aes(x='colonne', y='value')
    ) +
    p9.geom_point(
        descr_df_long.filter(pl.col('statistic') == '1%'),
        p9.aes(x='colonne', y='value')
    ) +    
    p9.geom_point(
        descr_df_long.filter(pl.col('statistic') == '5%'),
        p9.aes(x='colonne', y='value')
    ) +    
    p9.geom_segment(
        descr_df_long.filter(pl.col('statistic') == '25%'),
        p9.aes(
            x='colonne', 
            y='value', 
            xend='colonne', 
            yend=descr_df_long.filter(pl.col('statistic') == '75%')['value']
            ),
            color="black"
    ) + 
    p9.geom_point(
        descr_df_long.filter(pl.col('statistic') == '25%'),
        p9.aes(x='colonne', y='value')
    ) + 
    p9.geom_point(
        descr_df_long.filter(pl.col('statistic') == 'median/mode'),
        p9.aes(x='colonne', y='value')
    ) + 
    p9.geom_point(
        descr_df_long.filter(pl.col('statistic') == '75%'),
        p9.aes(x='colonne', y='value')
    ) + 
    p9.geom_point(
        descr_df_long.filter(pl.col('statistic') == '95%'),
        p9.aes(x='colonne', y='value')
    ) + 
    p9.geom_point(
        descr_df_long.filter(pl.col('statistic') == '99%'),
        p9.aes(x='colonne', y='value')
    ) +
    p9.coord_flip() +
    p9.theme_matplotlib()
)


# %%
# Variable to keep
col_to_keep = (
    pl.from_dicts(
        [{"trans_id":0,"trans_date":0,"trans_year":0,"trans_month":0,"trans_type_code":0,
        "trans_type_label":0,"price":0,"prop_type":0,"prop_year_harm":0,"prop_loc_dep":0,
        "prop_loc_citycode":0,"prop_loc_x":0,"prop_loc_y":0,"dist_tosea":0,"dist_tosea_corr":0,
        "n_floors":1,"n_bath":1,"n_show":1,"n_sink":1,"n_wc":1,"n_mrooms":1,
        "n_eatr":1,"n_slr":1,"n_kit8":1,"n_kit9":1,"n_washr":1,"n_ancrooms":1,
        "n_rooms":1,"farea":1,"has_water":0,"has_elec":0,"stair":0,"has_gas":0,
        "has_elevator":0,"has_cheating":0,"has_rchute":0,"has_mdrainage":0,"nth_floor":1,"s_land_artif":1,
        "s_land_agri":1,"s_land_nat":1,"n_garage":1,"n_pool":1,
        "n_terrace":1,"n_attic":1,"n_basmt":1,"n_otherannex":1}]
    )
    .unpivot()
    .filter(pl.col("value") == 1)
    .select("variable")
    .to_series()
    .to_list()
)
col_to_keep
# %%
#| label : stat_des_details_flat
(
    p9.ggplot(
        data
        .select(col_to_keep)
        .unpivot()
        .group_by("variable", "value")
        .agg((pl.len()/1000000).alias("count"))
        .filter(pl.col("value")<10)
        .cast({"value":pl.String}),
        p9.aes(y="count", x="variable", fill="value")
    ) +
    p9.geom_col(position=p9.position_stack(reverse=True)) +
    p9.theme_minimal() +
    p9.coord_flip()
)

# %%
#| label: stat_des_details_surf_flat_num
# (
#     data
#         .select([
#             "s_land_artif",
#             "s_land_nat",
#             "s_land_agri"
#         ])
#         .unpivot()
#         .group_by("variable", "value")
#         .agg(pl.len())
#         .filter(pl.col("value")<10)
#         .sort("variable", "value")
# )
# %%
#| label : stat_des_details_surf_flat_plot

(
    p9.ggplot(
        data
        .select([
            "s_land_artif",
            "s_land_nat",
            "s_land_agri"
        ])
        .unpivot()
        .filter(pl.col("value")>0),
        p9.aes("value")
    ) +
    p9.geom_histogram(bins = 25, fill='skyblue', color='black')  +
    p9.facet_wrap('~variable', scales='free') +
    p9.theme_minimal()
)


# %%
# Plot price (log) - seems ok 
(
    p9.ggplot(data.with_columns(
                price_log=pl.col("price").log(base=10)
            ).select(["price_log"]).unpivot(value_name="log_price"), 
            p9.aes(x='log_price')
    ) +
    p9.geom_histogram(bins = 100, fill='skyblue', color='black') +
    p9.facet_grid('~variable', scales='free') +
    p9.theme_minimal()
)
# %%
#|: label : stat_des_some_vars_pairplot
sns.pairplot(
    data=data_all.sample(1000).select(["prop_type", "n_mrooms", "farea", "n_slr", pl.col("price").log().floor(), "n_pool"]).cast({"prop_type":pl.Int32}).to_pandas(),
    vars=["prop_type", "n_mrooms", "farea", "n_slr","n_pool"],
    hue="price",
)
matplotlib.pyplot.show()
# %%
data.len()
# %%
#| label: stat_des75_pandas
# %%
import matplotlib.pyplot as plt
import pandas as pd

ax = data_75.hist(log=True, figsize=(12, 10), xlabelsize=8,    # x-axis label size
    ylabelsize=8)

# Set label sizes for all subplots
for axes in ax.flatten():
    axes.tick_params(axis='both', labelsize=8)  # Tick labels
    axes.set_xlabel(axes.get_xlabel(), fontsize=8)  # x-axis label
    axes.set_ylabel(axes.get_ylabel(), fontsize=8)  # y-axis label
    axes.set_title(axes.get_title(), fontsize=8)    # Subplot title

plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.suptitle("Custom Histogram", fontsize=12)  # Main title
plt.show()
