import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def remove_irrelevant(df):
    """
    Remove irrelevant columns from the dataframe
    :param df: Raw dataframe
    :return: df: Dataframe with irrelevant values removed
    """
    df.drop('source_url', axis=1, inplace=True)
    df.drop('incident_url', axis=1, inplace=True)
    df.drop('incident_url_fields_missing', axis=1, inplace=True)
    df.drop('sources', axis=1, inplace=True)
    return df


def map_to_numbers(df):
    """
    Fill empty values and encode string values to numbers in the dataframe
    :param df: Dataframe with string and/or empty values
    :return: df: Dataframe with numerical values
    """
    label_encoder = LabelEncoder()
    for each_col in df.columns:
        df[each_col].fillna(method='ffill', inplace=True)
        df[each_col].fillna(method='bfill', inplace=True)
        df[each_col] = label_encoder.fit_transform(df[each_col].astype(str))
    return df


def process_dfs(train_df, test_df):
    """
    Process dataframes for performing ML algorithm
    :param
        train_df: Training dataframe,
        test_df: Test dataframe
    :return:
        train_df: Processed training dataframe,
        test_df: Processed test dataframe,
        incident_ids: Incident IDs as series
    """
    test_df.drop("state", axis=1, inplace=True)
    train_df.drop("incident_id", axis=1, inplace=True)
    incident_ids = pandas.Series(data=test_df["incident_id"], name="incident_id")
    test_df.drop("incident_id", axis=1, inplace=True)
    return train_df, test_df, incident_ids


def get_knn_predictions(train_df, test_df):
    """
    Get predicted US states for each attack using nearest neighbour algorithm
    :param
        train_df: Training dataframe,
        test_df: Test dataframe
    :return: results: Array of predicted US states
    """
    nearest_neighbour = KNeighborsClassifier()
    nearest_neighbour.fit(train_df.drop("state", axis=1), train_df["state"])
    results = nearest_neighbour.predict(test_df)
    return results


def save_results_to_file(incident_ids, results):
    """
    Create dataframe of incident IDs and predicted US states and print to CSV file
    :param
        incident_ids: Incident IDs as series,
        results: Array of predict US states
    """
    final_df = pandas.DataFrame({"incident_id": incident_ids, "state": results})
    final_df.to_csv("results.csv", index=False, header=True)


def my_main(data_file_name):
    """
    Main function where functions are called from
    :param data_file_name: Name of file containing data regarding US gun violence
    """
    """Raw dataframe read from CSV file"""
    raw_df = pandas.read_csv(data_file_name, encoding="ISO-8859-1", low_memory=False)

    """Removed irrelevant values"""
    vals_removed_df = remove_irrelevant(raw_df)

    """Dataframe columns mapped to numbers"""
    mapped_df = map_to_numbers(vals_removed_df)

    """Split dataframe into training and testing dataframes"""
    train, test = train_test_split(mapped_df, test_size=0.25)
    test_df = test.copy()
    train_df = train.copy()

    """Process dataframes and get incident IDs as series"""
    train_df, test_df, incident_ids = process_dfs(train_df, test_df)

    """Array of predicted US states for each attack"""
    results = get_knn_predictions(train_df, test_df)

    """Save prediction results to CSV file"""
    save_results_to_file(incident_ids, results)


if __name__ == '__main__':
    file_name = "gun_violence.csv"
    my_main(file_name)
