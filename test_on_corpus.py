import random
import os
import pandas as pd
import re

from clean_test import prepare_training_set , vectorise_dataset, \
    train_all_models, plot_results


def results_for_autors(data_dir, aut_pair):

    train_fn1 = os.path.join(data_dir, aut_pair, "class1_training/class1_training.txt")
    train_fn2 = os.path.join(data_dir, aut_pair, "class2_training/class2_training.txt")
    X_train, y_train = prepare_training_set(train_fn1, train_fn2)

    test_fn1 = os.path.join(data_dir, aut_pair, "class1_test/class1_test.txt")
    test_fn2 = os.path.join(data_dir, aut_pair, "class2_test/class2_test.txt")

    X_test, y_test = prepare_training_set(test_fn1, test_fn2)
    X_train, X_test = vectorise_dataset(X_train, X_test, y_train)

    results = train_all_models(X_train, y_train, X_test, y_test)
    return results


def avg_result(all_results):
    results_dfs = [pd.DataFrame(r) for r in all_results]
    df_concat = pd.concat(results_dfs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    model_names = results_dfs[0].loc[:, 0]
    avg_res_df = pd.concat([model_names, df_means], axis=1)
    avg_res = [tuple(x) for x in avg_res_df.values]
    return avg_res


def main():
    random.seed(42)

    data_dir = "/home/rauf/Programs/shpilman/data"
    r = re.compile('.*-vs-.*')
    versus_dirs = list(filter(r.match, os.listdir(data_dir)))
    # versus_dirs = ['pratchett-vs-baxter', 'perumov-vs-lukjarenko', 'gaiman-vs-pratchett']
    print("Available datasets: %s" % versus_dirs)

    all_results = [results_for_autors(data_dir, aut_pair) for aut_pair in versus_dirs]
    avg_res = avg_result(all_results)
    plot_results(avg_res)

if __name__ == "__main__":
    main()
