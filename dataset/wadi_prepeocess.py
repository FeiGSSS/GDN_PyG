import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# code adapted from https://github.com/huankoh/CST-GL/blob/main/generate_data/generate_wadi_data.ipynb

def get_attack_times() -> list:
    attacks = []

    # Attack 1 
    start = datetime(2017,10,9,19,25,00)
    end = datetime(2017,10,9,19,50,16)
    attacks.append([start, end])
    
    # Attack 2
    start = datetime(2017,10,10,10,24,10)
    end = datetime(2017,10,10,10,34,00)
    attacks.append([start, end])

    # Attack 3-4
    start = datetime(2017,10,10,10,55,00)
    end = datetime(2017,10,10,11,24,00)
    attacks.append([start, end])

    # Attack 5
    start = datetime(2017,10,10,11,30,40)
    end = datetime(2017,10,10,11,44,50)
    attacks.append([start, end])
    
    # Attack 6
    start = datetime(2017,10,10,13,39,30)
    end = datetime(2017,10,10,13,50,40)
    attacks.append([start, end])

    # Attack 7
    start = datetime(2017,10,10,14,48,17)
    end = datetime(2017,10,10,14,59,55)
    attacks.append([start, end])


    # Attack 8
    start = datetime(2017,10,10,17,40,00)
    end = datetime(2017,10,10,17,49,40)
    attacks.append([start, end])


    # Attack 9
    start = datetime(2017,10,10,10,55,00)
    end = datetime(2017,10,10,10,56,27)
    attacks.append([start, end])

    # Attack 10
    start = datetime(2017,10,11,11,17,54)
    end = datetime(2017,10,11,11,31,20)
    attacks.append([start, end])


    # Attack 11
    start = datetime(2017,10,11,11,36,31)
    end = datetime(2017,10,11,11,47,00)
    attacks.append([start, end])


    # Attack 12
    start = datetime(2017,10,11,11,59,00)
    end = datetime(2017,10,11,12,5,00)
    attacks.append([start, end])


    # Attack 13
    start = datetime(2017,10,11,12,7,30)
    end = datetime(2017,10,11,12,10,52)
    attacks.append([start, end])

    # Attack 14
    start = datetime(2017,10,11,12,16,00)
    end = datetime(2017,10,11,12,25,36)
    attacks.append([start, end])

    # Attack 15
    start = datetime(2017,10,11,15,26,30)
    end = datetime(2017,10,11,15,37,00)
    attacks.append([start, end])
    
    return attacks


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

# Function to preprocess dataset
def preprocess_data(file_path, is_train=True, attack_times=None):
    # Read and preprocess data
    if is_train:
        data = pd.read_csv(file_path, header=3)
    else:
        data = pd.read_csv(file_path)
    data.drop(columns=['Row'], inplace=True)

    # Format datetime
    data['Date'] = data['Date'].apply(lambda x: '/'.join([i.zfill(2) for i in x.split('/')]))
    data['Time'] = data['Time'].apply(lambda x: x.replace('.000','')) 
    data['Time'] = data['Time'].apply(lambda x: ':'.join([i.zfill(2) for i in x.split(':')]))
    data['datetime'] = data.apply(lambda x: datetime.strptime(x.Date +' '+x.Time,'%m/%d/%Y %I:%M:%S %p'),axis=1)
    
    # Filter out columns of interest 
    data = data[['datetime'] + [i for i in data.columns if i not in ['Date','Time','datetime']]]
    data = data.sort_values('datetime')
    
    # handle missing values
    empty_cols = [col for col in data.columns if data[col].isnull().all()]
    data[empty_cols] = data[empty_cols].fillna(0)
    for i in data.columns[data.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        data[i].fillna(data[i].mean(), inplace=True)
        
    # Rename columns 
    pat = re.escape('\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\')
    rename_col = [re.sub(pat,'', i) for i in data.columns]
    data.columns = rename_col
    
    # add attack label
    if is_train:
        # add all 0 for training data
        data['attack'] = 0
    else:
        # add attack label for test data
        data['attack'] = data['datetime'].apply(lambda x: int(x in attack_times))
    
    # drop datetime column
    data = data.drop(columns=['datetime'])
    
    # Trim column names to remove potential leading/trailing spaces
    data = data.rename(columns=lambda x: x.strip())
    
    return data

def main():

    # Define file paths
    train_path = "./dataset/raw_data/wadi/WADI_14days.csv"
    test_path = "./dataset/raw_data/wadi/WADI_attackdata.csv"
    output_dir = "./dataset/processed_data/wadi/"

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define attack times for test data
    attacks = get_attack_times()
    attack_times = set()
    for start, end in attacks:
        attack_times.update([start + timedelta(seconds=i) for i in range(int((end - start).total_seconds()) + 1)])

    # Process and save training data
    print("Processing training data...")
    train_data = preprocess_data(train_path, is_train=True)
    test_data = preprocess_data(test_path, is_train=False, attack_times=attack_times)
    
    train_labels = train_data['attack']
    test_labels = test_data['attack']
    train_data = train_data.drop(columns=['attack'])
    test_data = test_data.drop(columns=['attack'])
    
    # Normalize the training and testing data
    print("Normalizing data...")
    x_train, x_test = normalize_data(train_data, test_data)
    # Update the train and test dataframes with normalized data
    for i, col in enumerate(train_data.columns):
        train_data[col] = x_train[:, i]
        test_data[col] = x_test[:, i]
    
    # Downsample the data
    print("Downsampling data...")
    d_train_x, d_train_labels = downsample_data(train_data, train_labels, 10)
    d_test_x, d_test_labels = downsample_data(test_data, test_labels, 10)
    
    # Create dataframes from downsampled data
    train_df = pd.DataFrame(d_train_x, columns=train_data.columns)
    test_df = pd.DataFrame(d_test_x, columns=test_data.columns)
    train_df['attack'] = d_train_labels
    test_df['attack'] = d_test_labels
    
    # Drop the first 2160 rows from the training dataframe
    print("Dropping first 2160 rows from training data...")
    train_df = train_df.iloc[2160:]
    
    # Save the processed data to new CSV files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Write the list of column names to a text file
    with open(os.path.join(output_dir, 'list.txt'), 'w') as f:
        for col in train_data.columns:
            f.write(col + '\n')
    print("Done!")
    
    print(train_df.shape)
    print(test_df.shape)
    print(train_df.head())
    print(test_df.head())

if __name__ == '__main__':
    main()