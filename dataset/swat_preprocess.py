import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler

def downsample_data(data, labels, target_length):
    """
    Downsamples a dataset to a specified length by taking the median over equally sized windows.

    :param data: A list of lists or a 2D numpy array, where each inner list is a data sample.
    :param labels: A list of labels corresponding to the data samples.
    :param target_length: The target downsampling length.
    :return: A tuple containing the downsampled data and labels.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    if data.shape[0] != len(labels):
        raise ValueError("Length of data and labels must be the same.")

    num_samples, num_features = data.shape
    downsample_factor = num_samples // target_length

    if downsample_factor == 0:
        raise ValueError("Target length is greater than the number of samples.")

    # Downsampling data
    reshaped_data = data[:downsample_factor * target_length].reshape(-1, target_length, num_features)
    downsampled_data = np.median(reshaped_data, axis=1)

    # Downsampling labels
    reshaped_labels = labels[:downsample_factor * target_length].reshape(-1, target_length)
    downsampled_labels = np.round(np.max(reshaped_labels, axis=1))

    return downsampled_data.tolist(), downsampled_labels.tolist()

def normalize_data(training_data, testing_data):
    """
    Normalize the training and testing data to a range of [0, 1].

    :param training_data: A numpy array or a pandas dataframe representing the training data.
    :param testing_data: A numpy array or a pandas dataframe representing the testing data.
    :return: A tuple containing the normalized training and testing data.
    """
    if not isinstance(training_data, (np.ndarray, pd.DataFrame)) or not isinstance(testing_data, (np.ndarray, pd.DataFrame)):
        raise TypeError("Input data should be a numpy array or a pandas dataframe.")

    # Initialize the MinMaxScaler to scale data between 0 and 1
    normalizer = MinMaxScaler(feature_range=(0, 1))

    # Fit the normalizer to the training data and transform both training and testing data
    normalized_training_data = normalizer.fit_transform(training_data)
    normalized_testing_data = normalizer.transform(testing_data)

    return normalized_training_data, normalized_testing_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess the SWaT dataset.')
    parser.add_argument('--train_path', type=str, default='./dataset/raw_data/swat/SWaT_Dataset_Normal_v0.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test_path', type=str, default='./dataset/raw_data/swat/SWaT_Dataset_Attack_v0.csv',
                        help='Path to the testing data CSV file.')
    parser.add_argument('--output_path', type=str, default='./dataset/processed_data/swat/',
                        help='Path to the output directory.')
    args = parser.parse_args()
    
    
    # Read data from CSV files
    print("Reading data from CSV files...")
    train = pd.read_csv(args.train_path, low_memory=False, header=1).iloc[:, 1:]
    test = pd.read_csv(args.test_path, low_memory=False).iloc[:, 1:]
    
    # rename column 'Normal/Attack' to 'attack'
    train = train.rename(columns={'Normal/Attack': 'attack'})
    test = test.rename(columns={'Normal/Attack': 'attack'})
    
    # replace 'Normal' with 0 and 'Attack' with 1
    train['attack'] = train['attack'].replace(['Normal', 'Attack'], [0, 1])
    test['attack'] = test['attack'].replace(['Normal', 'Attack', 'A ttack'], [0, 1, 1])
    
    print("Handling missing values...")
    train = train.fillna(train.mean()).fillna(0)
    test = test.fillna(test.mean()).fillna(0)

    # Trim column names to remove potential leading/trailing spaces
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    # Separate labels
    train_labels = train['attack']
    test_labels = test['attack']
    train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    # Normalize the training and testing data
    print("Normalizing data...")
    x_train, x_test = normalize_data(train, test)

    # Update the train and test dataframes with normalized data
    for i, col in enumerate(train.columns):
        train[col] = x_train[:, i]
        test[col] = x_test[:, i]

    # Downsample the data
    print("Downsampling data...")
    d_train_x, d_train_labels = downsample_data(train, train_labels, 10)
    d_test_x, d_test_labels = downsample_data(test, test_labels, 10)

    # Create dataframes from downsampled data
    train_df = pd.DataFrame(d_train_x, columns=train.columns)
    test_df = pd.DataFrame(d_test_x, columns=test.columns)
    train_df['attack'] = d_train_labels
    test_df['attack'] = d_test_labels

    # Drop the first 2160 rows from the training dataframe
    print("Dropping first 2160 rows from training data...")
    train_df = train_df.iloc[2160:]

    # Save the processed data to new CSV files
    train_df.to_csv(os.path.join(args.output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_path, 'test.csv'), index=False)

    # Write the list of column names to a text file
    with open(os.path.join(args.output_path, 'list.txt'), 'w') as f:
        for col in train.columns:
            f.write(col + '\n')
    print("Done!")
    
    print(train_df.shape)
    print(test_df.shape)
    print(train_df.head())
    print(test_df.head())

if __name__ == "__main__":
    main()
