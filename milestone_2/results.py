import pandas as pd


# method to calculate scores for classification
def print_result(predictions, y_test):
    # calculate and print confusion matrix
    y_actual = pd.Series(predictions)
    y_expected = pd.Series(y_test)
    conf_matrix = pd.crosstab(y_expected, y_actual).values
    print("confusion matrix:")
    print(conf_matrix, "\n")

    # calculate precision for all sentiment classes
    conf_matrix = conf_matrix

    sum_rows = conf_matrix.sum(axis=0)
    sum_cols = conf_matrix.sum(axis=1)

    precision_a = round(conf_matrix[0][0] / sum_rows[0], 2)
    precision_b = round(conf_matrix[1][1] / sum_rows[1], 2)
    precision_c = round(conf_matrix[2][2] / sum_rows[2], 2)
    weighted_avg_precision = ((precision_a * sum_cols[0]) + (precision_b * sum_cols[1]) + (precision_c * sum_cols[2])) / len(y_test)

    # calculate recall for all sentiment classes
    recall_a = round(conf_matrix[0][0] / sum_cols[0], 2)
    recall_b = round(conf_matrix[1][1] / sum_cols[1], 2)
    recall_c = round(conf_matrix[2][2] / sum_cols[2], 2)
    weighted_avg_recall = ((recall_a * sum_cols[0]) + (recall_b * sum_cols[1]) + (recall_c * sum_cols[2])) / len(y_test)

    # calculate f1-score for all sentiment classes
    f_score_a = round(2 * precision_a * recall_a / (precision_a + recall_a), 2)
    f_score_b = round(2 * precision_b * recall_b / (precision_b + recall_b), 2)
    f_score_c = round(2 * precision_c * recall_c / (precision_c + recall_c), 2)
    weighted_avg_f1 = ((f_score_a * sum_cols[0]) + (f_score_b * sum_cols[1]) + (f_score_c * sum_cols[2])) / len(y_test)

    # print table containing precision, recall and f1-score for all sentiment classes
    d = {0: [precision_a, recall_a, f_score_a], 1: [precision_b, recall_b, f_score_b],
         2: [precision_c, recall_c, f_score_c], 'wt_avg': [round(weighted_avg_precision, 2), round(weighted_avg_recall, 2), round(weighted_avg_f1, 2)]}
    print("{:<10} {:<10} {:<10} {:<10}".format(' ', 'Precision', 'Recall', 'f1-score'))
    for k, v in d.items():
        p, r, f = v
        print("{:<10} {:<10} {:<10} {:<10}".format(k, p, r, f))

    # calculate and print total accuracy
    accuracy = round((conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2]) / len(predictions), 2)
    print("accuracy: ", accuracy)
