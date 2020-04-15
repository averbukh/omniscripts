# coding: utf-8
import os
import sys
import time
import traceback
import warnings
from timeit import default_timer as timer

import ibis

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    cod,
    compare_dataframes,
    import_pandas_into_module_namespace,
    load_data_pandas,
    mse,
    print_results,
)

warnings.filterwarnings("ignore")

def pandas_original():
    import pandas as pd
    import xgboost as xgb

    global x_train_pandas_original
    global y_train_pandas_original
    global x_valid_pandas_original
    global y_valid_pandas_original

    PATH = '/localdisk/benchmark_datasets/santander'

    train_pd = pd.read_csv('%s/train.csv'%PATH)

    for i in range(200):
        col = 'var_%d'%i
        var_count = train_pd.groupby(col).agg({col:'count'})

        var_count.columns = ['%s_count'%col]
        var_count = var_count.reset_index()

        train_pd = train_pd.merge(var_count,on=col,how='left')

    for i in range(200):
        col = 'var_%d'%i

        mask = train_pd['%s_count'%col]>1

        train_pd.loc[mask,'%s_gt1'%col] = train_pd.loc[mask,col]

    # train, test data split
    train,valid = train_pd[:-10000],train_pd[-10000:]
    x_train_pandas_original = train.drop(['target','ID_code'],axis=1)
    y_train_pandas_original = train['target']
    x_valid_pandas_original = valid.drop(['target','ID_code'],axis=1)
    y_valid_pandas_original = valid['target']

    xgb_params = {
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'max_depth': 1,
                'nthread':56,
                'eta':0.1,
                'silent':1,
                'subsample':0.5,
                'colsample_bytree': 0.05,
                'eval_metric':'auc',
    }
    dtrain = xgb.DMatrix(data=x_train_pandas_original, label=y_train_pandas_original)
    dvalid = xgb.DMatrix(data=x_valid_pandas_original, label=y_valid_pandas_original)

    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    clf = xgb.train(xgb_params, dtrain=dtrain,
                    num_boost_round=10000, evals=watchlist,
                    early_stopping_rounds=30, maximize=True,
                    verbose_eval=1000)

    yp = clf.predict(dvalid)

    score_mse = mse(y_valid_pandas_original, yp)
    score_cod = cod(y_valid_pandas_original, yp)

    print('[pandas_original] Scores: ')
    print('  mse = ', score_mse)
    print('  cod = ', score_cod)

def compare_with_pandas_original(title, pandas_df, ibis_df):
    import math

    print('[compare_with_pandas_original]', title)

    column_count = pandas_df.shape[1]
    row_count = pandas_df.shape[0]

    assert column_count == ibis_df.shape[1]
    assert row_count == ibis_df.shape[0]

    # pandas_df.columns = pandas_df.columns.str.strip()
    # ibis_df.columns = ibis_df.columns.str.strip()
    pdf = pandas_df.sort_values(by=list(pandas_df.columns))
    idf = ibis_df.sort_values(by=list(pandas_df.columns))

    # to_idf_map = []
    # idf_row = 0
    # for pdf_row in range(row_count):
    #     idf_id = None
    #     for pandas_col_name in pandas_df.columns:
    #         v = pdf.iloc[pdf_row, ]

    mins = {}
    maxes = {}
    sums = {}
    sq_sums = {}
    rel_mins = {}
    rel_maxes = {}
    rel_sums = {}
    rel_sq_sums = {}
    for pandas_col_name in pandas_df.columns:
        pandas_col = pdf.loc[:, pandas_col_name]
        ibis_col = idf.loc[:, pandas_col_name]
        # print(pandas_col_name, ':', pandas_col.shape, '; ', ibis_col.shape)
        for row in range(row_count):
            pandas_value = pandas_col.iloc[row]
            if isinstance(pandas_value, float):
                ibis_value = ibis_col.iloc[row]
                delta = abs(pandas_value - ibis_value)
                rel_delta = abs(delta / pandas_value)

                if not math.isnan(delta):
                    if pandas_col_name not in sums:
                        sums[pandas_col_name] = delta
                    else:
                        sums[pandas_col_name] += delta

                    if pandas_col_name not in sq_sums:
                        sq_sums[pandas_col_name] = delta * delta
                    else:
                        sq_sums[pandas_col_name] += delta * delta

                    if pandas_col_name not in mins:
                        mins[pandas_col_name] = delta
                    else:
                        if delta < mins[pandas_col_name]:
                            mins[pandas_col_name] = delta

                    if pandas_col_name not in maxes:
                        maxes[pandas_col_name] = delta
                    else:
                        if delta > maxes[pandas_col_name]:
                            maxes[pandas_col_name] = delta

                if not math.isnan(rel_delta):
                    if pandas_col_name not in rel_sums:
                        rel_sums[pandas_col_name] = rel_delta
                    else:
                        rel_sums[pandas_col_name] += rel_delta

                    if pandas_col_name not in rel_sq_sums:
                        rel_sq_sums[pandas_col_name] = rel_delta * rel_delta
                    else:
                        rel_sq_sums[pandas_col_name] += rel_delta * rel_delta

                    if pandas_col_name not in rel_mins:
                        rel_mins[pandas_col_name] = rel_delta
                    else:
                        if rel_delta < rel_mins[pandas_col_name]:
                            rel_mins[pandas_col_name] = rel_delta

                    if pandas_col_name not in rel_maxes:
                        rel_maxes[pandas_col_name] = rel_delta
                    else:
                        if rel_delta > rel_maxes[pandas_col_name]:
                            rel_maxes[pandas_col_name] = rel_delta

    print('mins:', mins)
    print('maxes:', maxes)
    print('sums:', sums)
    print('sq_sums:', sq_sums)
    print('rel_mins:', rel_mins)
    print('rel_maxes:', rel_maxes)
    print('rel_sums:', rel_sums)
    print('rel_sq_sums:', rel_sq_sums)
    print('row count:', row_count)

def compare_all_with_pandas_original():
    import pandas as pd

    global x_train_pandas_original
    global y_train_pandas_original
    global x_valid_pandas_original
    global y_valid_pandas_original

    global x_train_ibis_original
    global y_train_ibis_original
    global x_valid_ibis_original
    global y_valid_ibis_original

    concatenated_x_pandas_original = pd.concat([x_train_pandas_original, x_valid_pandas_original])
    concatenated_x_ibis_original = pd.concat([x_train_ibis_original, x_valid_ibis_original])

    # y_train_ibis_original = y_train_ibis_original.rename(columns={"target0": "target"})
    # y_valid_ibis_original = y_valid_ibis_original.rename(columns={"target0": "target"})
    # y_train_ibis_original = y_train_ibis_original.rename("target")
    # y_valid_ibis_original = y_valid_ibis_original.rename("target")

    compare_with_pandas_original('concatenated_x_pandas_original vs concatenated_x_ibis_original', concatenated_x_pandas_original, concatenated_x_ibis_original)

# Dataset link
# https://www.kaggle.com/c/santander-customer-transaction-prediction/data

# Current script prerequisites:
# 1) Patched OmniSci version (https://github.com/intel-go/omniscidb/tree/ienkovich/santander)
# 2) Patched Ibis version (https://github.com/intel-go/ibis/tree/develop)


def etl_pandas(filename, columns_names, columns_types, etl_keys):
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = timer()
    train_pd = load_data_pandas(
        filename=filename,
        columns_names=columns_names,
        columns_types=columns_types,
        header=0,
        nrows=None,
        use_gzip=filename.endswith(".gz"),
        pd=run_benchmark.__globals__["pd"],
    )
    etl_times["t_readcsv"] = round((timer() - t0) * 1000)

    t_etl_begin = timer()

    for i in range(200):
        col = "var_%d" % i
        var_count = train_pd.groupby(col).agg({col: "count"})

        var_count.columns = ["%s_count" % col]
        var_count = var_count.reset_index()

        train_pd = train_pd.merge(var_count, on=col, how="left")

    for i in range(200):
        col = "var_%d" % i

        mask = train_pd["%s_count" % col] > 1
        train_pd.loc[mask, "%s_gt1" % col] = train_pd.loc[mask, col]

    train_pd = train_pd.drop(["ID_code"], axis=1)
    etl_times["t_etl"] = round((timer() - t_etl_begin) * 1000)

    return train_pd, etl_times


def etl_ibis(
    filename,
    columns_names,
    columns_types,
    database_name,
    table_name,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
    ipc_connection,
    validation,
    run_import_queries,
    etl_keys,
    import_mode,
):
    tmp_table_name = "tmp_table"

    etl_times = {key: 0.0 for key in etl_keys}

    omnisci_server_worker.create_database(database_name, delete_if_exists=delete_old_database)

    if run_import_queries:
        etl_times_import = {
            "t_readcsv_by_ibis": 0.0,
            "t_readcsv_by_COPY": 0.0,
            "t_readcsv_by_FSI": 0.0,
        }

        # SQL statemnts preparation for data file import queries
        connect_to_db_sql_template = "\c {0} admin HyperInteractive"
        create_table_sql_template = """
        CREATE TABLE {0} ({1});
        """
        import_by_COPY_sql_template = """
        COPY {0} FROM '{1}' WITH (header='{2}');
        """
        import_by_FSI_sql_template = """
        CREATE TEMPORARY TABLE {0} ({1}) WITH (storage_type='CSV:{2}');
        """
        drop_table_sql_template = """
        DROP TABLE IF EXISTS {0};
        """

        import_query_cols_list = (
            ["ID_code TEXT ENCODING NONE, \n", "target SMALLINT, \n"]
            + ["var_%s DOUBLE, \n" % i for i in range(199)]
            + ["var_199 DOUBLE"]
        )
        import_query_cols_str = "".join(import_query_cols_list)

        create_table_sql = create_table_sql_template.format(tmp_table_name, import_query_cols_str)
        import_by_COPY_sql = import_by_COPY_sql_template.format(tmp_table_name, filename, "true")
        import_by_FSI_sql = import_by_FSI_sql_template.format(
            tmp_table_name, import_query_cols_str, filename
        )

        # data file import by ibis
        columns_types_import_query = ["string", "int64"] + ["float64" for _ in range(200)]
        schema_table_import = ibis.Schema(names=columns_names, types=columns_types_import_query)
        omnisci_server_worker.create_table(
            table_name=tmp_table_name, schema=schema_table_import, database=database_name,
        )

        table_import_query = omnisci_server_worker.database(database_name).table(tmp_table_name)
        t0 = timer()
        table_import_query.read_csv(filename, delimiter=",")
        etl_times_import["t_readcsv_by_ibis"] = round((timer() - t0) * 1000)

        # data file import by FSI
        omnisci_server_worker.drop_table(tmp_table_name)
        t0 = timer()
        omnisci_server_worker.execute_sql_query(import_by_FSI_sql)
        etl_times_import["t_readcsv_by_FSI"] = round((timer() - t0) * 1000)

        omnisci_server_worker.drop_table(tmp_table_name)

        # data file import by SQL COPY statement
        omnisci_server_worker.execute_sql_query(create_table_sql)

        t0 = timer()
        omnisci_server_worker.execute_sql_query(import_by_COPY_sql)
        etl_times_import["t_readcsv_by_COPY"] = round((timer() - t0) * 1000)

        omnisci_server_worker.drop_table(tmp_table_name)

        etl_times.update(etl_times_import)

    if create_new_table:
        # Create table and import data for ETL queries
        schema_table = ibis.Schema(names=columns_names, types=columns_types)
        if import_mode == "copy-from":
            omnisci_server_worker.create_table(
                table_name=table_name, schema=schema_table, database=database_name,
            )
            table_import = omnisci_server_worker.database(database_name).table(table_name)

            t0 = timer()
            table_import.read_csv(filename, header=True, quotechar="", delimiter=",")
            etl_times["t_readcsv"] = round((timer() - t0) * 1000)

        elif import_mode == "pandas":
            # Datafiles import
            columns_types_converted = [
                "float64" if (x.startswith("decimal")) else x for x in columns_types
            ]
            t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis(
                table_name=table_name,
                data_files_names=filename,
                files_limit=1,
                columns_names=columns_names,
                columns_types=columns_types_converted,
                header=0,
                nrows=None,
                compression_type="gzip" if filename.endswith("gz") else None,
                validation=validation,
            )
            etl_times["t_readcsv"] = round((t_import_pandas + t_import_ibis) * 1000)

        elif import_mode == "fsi":
            try:
                unzip_name = None
                if filename.endswith("gz"):
                    import gzip

                    unzip_name = "/tmp/santander-fsi.csv"

                    with gzip.open(filename, "rb") as gz_input:
                        with open(unzip_name, "wb") as output:
                            output.write(gz_input.read())

                t0 = timer()
                omnisci_server_worker._conn.create_table_from_csv(
                    table_name, unzip_name or filename, schema_table
                )
                etl_times["t_readcsv"] = round((timer() - t0) * 1000)

            finally:
                if filename.endswith("gz"):
                    import os

                    os.remove(unzip_name)

    # Second connection - this is ibis's ipc connection for DML
    omnisci_server_worker.connect_to_server(database_name, ipc=ipc_connection)
    table = omnisci_server_worker.database(database_name).table(table_name)

    # group_by/count, merge (join) and filtration queries
    # We are making 400 columns and then insert them into original table thus avoiding
    # nested sql requests
    t_etl_start = timer()
    count_cols = []
    orig_cols = ["ID_code", "target"] + ["var_%s" % i for i in range(200)]
    cast_cols = []
    cast_cols.append(table["target"].cast("int64").name("target0"))
    gt1_cols = []
    for i in range(200):
        col = "var_%d" % i
        col_count = "var_%d_count" % i
        col_gt1 = "var_%d_gt1" % i
        w = ibis.window(group_by=col)
        count_cols.append(table[col].count().over(w).name(col_count))
        gt1_cols.append(
            ibis.case()
            .when(table[col].count().over(w).name(col_count) > 1, table[col].cast("float32"),)
            .else_(ibis.null())
            .end()
            .name("var_%d_gt1" % i)
        )
        cast_cols.append(table[col].cast("float32").name(col))

    table = table.mutate(count_cols)
    table = table.drop(orig_cols)
    table = table.mutate(gt1_cols)
    table = table.mutate(cast_cols)

    table_df = table.execute()

    etl_times["t_etl"] = round((timer() - t_etl_start) * 1000)
    return table_df, etl_times


def split_step(data, target):
    t0 = timer()
    train, valid = data[:-10000], data[-10000:]
    split_time = round((timer() - t0) * 1000)

    x_train = train.drop([target], axis=1)

    y_train = train[target]

    x_test = valid.drop([target], axis=1)

    y_test = valid[target]

    return (x_train, y_train, x_test, y_test), split_time


def ml(ml_data, target, ml_keys, ml_score_keys):
    import xgboost

    global x_train_ibis_original
    global y_train_ibis_original
    global x_valid_ibis_original
    global y_valid_ibis_original

    ml_times = {key: 0.0 for key in ml_keys}
    ml_scores = {key: 0.0 for key in ml_score_keys}

    (x_train, y_train, x_test, y_test), ml_times["t_train_test_split"] = split_step(
        ml_data, target
    )

    if target == 'target0':
        x_train_ibis_original = x_train
        y_train_ibis_original = y_train
        x_valid_ibis_original = x_test
        y_valid_ibis_original = y_test

    t0 = timer()
    training_dmat_part = xgboost.DMatrix(data=x_train, label=y_train)
    testing_dmat_part = xgboost.DMatrix(data=x_test, label=y_test)
    ml_times["t_dmatrix"] = round((timer() - t0) * 1000)

    watchlist = [(testing_dmat_part, "eval"), (training_dmat_part, "train")]
    xgb_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": 1,
        "nthread": 56,
        "eta": 0.1,
        "silent": 1,
        "subsample": 0.5,
        "colsample_bytree": 0.05,
        "eval_metric": "auc",
    }

    t0 = timer()
    model = xgboost.train(
        xgb_params,
        dtrain=training_dmat_part,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=30,
        maximize=True,
        verbose_eval=1000,
    )
    ml_times["t_train"] = round((timer() - t0) * 1000)

    t0 = timer()
    yp = model.predict(testing_dmat_part)
    ml_times["t_inference"] = round((timer() - t0) * 1000)

    ml_scores["mse"] = mse(y_test, yp)
    ml_scores["cod"] = cod(y_test, yp)

    ml_times["t_ml"] += round(ml_times["t_train"] + ml_times["t_inference"])

    return ml_scores, ml_times


def run_benchmark(parameters):

    ignored_parameters = {
        "dfiles_num": parameters["dfiles_num"],
        "gpu_memory": parameters["gpu_memory"],
    }
    warnings.warn(f"Parameters {ignored_parameters} are irnored", RuntimeWarning)

    parameters["data_file"] = parameters["data_file"].replace("'", "")

    etl_times_ibis = None
    etl_times = None
    ml_times_ibis = None
    ml_times = None

    var_cols = ["var_%s" % i for i in range(200)]
    count_cols = ["var_%s_count" % i for i in range(200)]
    gt1_cols = ["var_%s_gt1" % i for i in range(200)]
    columns_names = ["ID_code", "target"] + var_cols
    columns_types_pd = ["object", "int64"] + ["float64" for _ in range(200)]
    columns_types_ibis = ["string", "int32"] + ["decimal(8, 4)" for _ in range(200)]

    etl_keys = ["t_readcsv", "t_etl"]
    ml_keys = ["t_train_test_split", "t_ml", "t_train", "t_inference", "t_dmatrix"]
    ml_score_keys = ["mse", "cod"]
    try:

        import_pandas_into_module_namespace(
            namespace=run_benchmark.__globals__,
            mode=parameters["pandas_mode"],
            ray_tmpdir=parameters["ray_tmpdir"],
            ray_memory=parameters["ray_memory"],
        )

        if not parameters["no_ibis"]:
            ml_data_ibis, etl_times_ibis = etl_ibis(
                filename=parameters["data_file"],
                run_import_queries=False,
                columns_names=columns_names,
                columns_types=columns_types_ibis,
                database_name=parameters["database_name"],
                table_name=parameters["table"],
                omnisci_server_worker=parameters["omnisci_server_worker"],
                delete_old_database=not parameters["dnd"],
                create_new_table=not parameters["dni"],
                ipc_connection=parameters["ipc_connection"],
                validation=parameters["validation"],
                etl_keys=etl_keys,
                import_mode=parameters["import_mode"],
            )

            print_results(results=etl_times_ibis, backend="Ibis", unit="ms")
            etl_times_ibis["Backend"] = "Ibis"

        ml_data, etl_times = etl_pandas(
            filename=parameters["data_file"],
            columns_names=columns_names,
            columns_types=columns_types_pd,
            etl_keys=etl_keys,
        )
        print_results(results=etl_times, backend=parameters["pandas_mode"], unit="ms")
        etl_times["Backend"] = parameters["pandas_mode"]

        if not parameters["no_ml"]:
            ml_scores, ml_times = ml(
                ml_data=ml_data, target="target", ml_keys=ml_keys, ml_score_keys=ml_score_keys,
            )
            print_results(results=ml_times, backend=parameters["pandas_mode"], unit="ms")
            ml_times["Backend"] = parameters["pandas_mode"]
            print_results(results=ml_scores, backend=parameters["pandas_mode"])
            ml_scores["Backend"] = parameters["pandas_mode"]

            if not parameters["no_ibis"]:
                ml_scores_ibis, ml_times_ibis = ml(
                    ml_data=ml_data_ibis,
                    target="target0",
                    ml_keys=ml_keys,
                    ml_score_keys=ml_score_keys,
                )
                print_results(results=ml_times_ibis, backend="Ibis", unit="ms")
                ml_times_ibis["Backend"] = "Ibis"
                print_results(results=ml_scores_ibis, backend="Ibis")
                ml_scores_ibis["Backend"] = "Ibis"

        # Results validation block (comparison of etl_ibis and etl_pandas outputs)
        if parameters["validation"] and not parameters["no_ibis"]:
            print("Validation of ETL query results with ...")
            cols_to_sort = ["var_0", "var_1", "var_2", "var_3", "var_4"]

            ml_data_ibis = ml_data_ibis.rename(columns={"target0": "target"})
            # compare_dataframes doesn't sort pandas dataframes
            ml_data.sort_values(by=cols_to_sort, inplace=True)

            compare_result = compare_dataframes(
                ibis_dfs=[ml_data_ibis], pandas_dfs=[ml_data], sort_cols=cols_to_sort, drop_cols=[]
            )

        pandas_original()
        compare_all_with_pandas_original()

        return {"ETL": [etl_times_ibis, etl_times], "ML": [ml_times_ibis, ml_times]}
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
