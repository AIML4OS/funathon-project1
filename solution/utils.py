import logging
import datetime


def setup_logging():
    """Configure logging with both console and file handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'funathon_aiml4os_project1_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    return logging.getLogger(__name__)


def set_seed():
    return 202605


def check_data(df):
    dataframe_name = [name for name, val in globals().items() if val is df][0]

    res_dict = {
        "dataframe_name": dataframe_name,
        "hash": int(pd.util.hash_pandas_object(df).sum()),
        "n_cols": df.shape[1],
        "n_rows": df.shape[0]
    }

    msg = f"""\
= {dataframe_name} ========= 
hash              : {res_dict["hash"]}
number of columns : {res_dict["n_cols"]} 
number of rows    : {res_dict["n_rows"]} 

"""

    res_dict["msg"] = msg

    return res_dict


# print(check_data(df)["msg"])
# print(check_data(X_train)["msg"])
# print(check_data(X_test)["msg"])

# = df ========= 
# hash              : 17566835255591554045
# number of columns : 40 
# number of rows    : 1694285 

# = X_train ========= 
# hash              : 9740245480247122702
# number of columns : 39 
# number of rows    : 1355428 

# = X_test ========= 
# hash              : 15218863447364480506
# number of columns : 39 
# number of rows    : 338857 
