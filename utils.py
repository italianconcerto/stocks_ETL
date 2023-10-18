
import os
from tqdm import tqdm
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import *


def get_stock_data(ticker, start, end, interval='1d'):
    end_date = (datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end_date, interval=interval)
    
    if len(data) == 0 or data.index[0].strftime('%Y-%m-%d') != start or data.index[-1].strftime('%Y-%m-%d') != end:
        return None
    
    data['Close_Open_Diff_3_days'] = data['Close'].shift(-2) - data['Close']
    data['Close_Open_Diff_5_days'] = data['Close'].shift(-4) - data['Close']
    data['Close_Open_Diff_7_days'] = data['Close'].shift(-6) - data['Close']

    # quantiles = data['Close_Open_Diff'].quantile([0.333, 0.666])
    # data['Quantile_3days'] = 0
    # data['Quantile_5days'] = 0
    # data['Quantile_7days'] = 0
    # data.loc[data['Close_Open_Diff'] <= quantiles.iloc[0], 'Quantile'] = 1
    # data.loc[data['Close_Open_Diff'] > quantiles.iloc[1], 'Quantile'] = 3
    # data.loc[(data['Close_Open_Diff'] > quantiles.iloc[0]) & (data['Close_Open_Diff'] <= quantiles.iloc[1]), 'Quantile'] = 2
    return data


def delete_files_in_folders(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {str(e)}")
                

def process_datasets(stocks):
    datasets = []
    for stock in tqdm(stocks):
        dataset = stock[0]
        stock_name = stock[1]

        dataset['Close_Open_Diff_3_days'] = dataset['Close'].shift(-2) - dataset['Close']
        dataset['Close_Open_Diff_5_days'] = dataset['Close'].shift(-4) - dataset['Close']
        dataset['Close_Open_Diff_7_days'] = dataset['Close'].shift(-6) - dataset['Close']

        dataset = dataset.sort_index()
        if stock_name in tech_companies:
            plot_distribution(dataset, 'Close_Open_Diff_3_days', stock_name)
            plot_distribution(dataset, 'Close_Open_Diff_5_days', stock_name)
            plot_distribution(dataset, 'Close_Open_Diff_7_days', stock_name)
        dataset['Quantile_33_3_days'] = dataset['Close_Open_Diff_3_days'].quantile(0.33)
        dataset['Quantile_66_3_days'] = dataset['Close_Open_Diff_3_days'].quantile(0.66)
        dataset['Quantile_33_5_days'] = dataset['Close_Open_Diff_5_days'].quantile(0.33)
        dataset['Quantile_66_5_days'] = dataset['Close_Open_Diff_5_days'].quantile(0.66)
        dataset['Quantile_33_7_days'] = dataset['Close_Open_Diff_7_days'].quantile(0.33)
        dataset['Quantile_66_7_days'] = dataset['Close_Open_Diff_7_days'].quantile(0.66)

        dataset['DiffLabel3Days'] = 0.0
        differences = []
        for i in range(0, len(dataset) - 12):
            # breakpoint()
            window = dataset.iloc[i+10:i+13]
            
            diff_value = (window.iloc[2]['Close'] - window.iloc[0]['Close'])
            differences.append(diff_value)
            dataset.loc[[i], 'DiffLabel3Days'] = diff_value
        
        
        dataset['DiffLabel5Days'] = 0.0
        differences = []
        for i in range(0, len(dataset) - 15):
            window = dataset.iloc[i+10:i+15]
            diff_value = (window.iloc[4]['Close'] - window.iloc[0]['Close'])
            differences.append(diff_value)
            dataset.loc[[i], 'DiffLabel5Days'] = diff_value

        dataset['DiffLabel7Days'] = 0.0
        differences = []
        for i in range(0, len(dataset) - 17):
            window = dataset.iloc[i+10:i+17]
            diff_value = (window.iloc[6]['Close'] - window.iloc[0]['Close'])
            differences.append(diff_value)
            dataset.loc[[i], 'DiffLabel7Days'] = diff_value

        
        dataset.loc[dataset['DiffLabel3Days'] > dataset['Quantile_66_3_days'], 'Label3Days'] = 3
        dataset.loc[dataset['DiffLabel3Days'] < dataset['Quantile_33_3_days'], 'Label3Days'] = 1
        dataset.loc[(dataset['DiffLabel3Days'] >= dataset['Quantile_33_3_days']) & (dataset['DiffLabel3Days'] <= dataset['Quantile_66_3_days']), 'Label3Days'] = 2

        dataset.loc[dataset['DiffLabel5Days'] > dataset['Quantile_66_5_days'], 'Label5Days'] = 3
        dataset.loc[dataset['DiffLabel5Days'] < dataset['Quantile_33_5_days'], 'Label5Days'] = 1
        dataset.loc[(dataset['DiffLabel5Days'] >= dataset['Quantile_33_5_days']) & (dataset['DiffLabel5Days'] <= dataset['Quantile_66_5_days']), 'Label5Days'] = 2

        dataset.loc[dataset['DiffLabel7Days'] > dataset['Quantile_66_7_days'], 'Label7Days'] = 3
        dataset.loc[dataset['DiffLabel7Days'] < dataset['Quantile_33_7_days'], 'Label7Days'] = 1
        dataset.loc[(dataset['DiffLabel7Days'] >= dataset['Quantile_33_7_days']) & (dataset['DiffLabel7Days'] <= dataset['Quantile_66_7_days']), 'Label7Days'] = 2

        # breakpoint()

        datasets.append(dataset)
        
    return datasets
        # dataset['Rolling_3D_Quantile_33'] = dataset['Close_Open_Diff'].rolling(window=3).quantile(0.33)
        # dataset['Rolling_3D_Quantile_66'] = dataset['Close_Open_Diff'].rolling(window=3).quantile(0.66)
        # dataset['Rolling_5D_Quantile_33'] = dataset['Close_Open_Diff'].rolling(window=5).quantile(0.33)
        # dataset['Rolling_5D_Quantile_66'] = dataset['Close_Open_Diff'].rolling(window=5).quantile(0.66)
        # dataset['Rolling_7D_Quantile_33'] = dataset['Close_Open_Diff'].rolling(window=7).quantile(0.33)
        # dataset['Rolling_7D_Quantile_66'] = dataset['Close_Open_Diff'].rolling(window=7).quantile(0.66)
        # for i in range(len(dataset) - 6):
        #     window = dataset.iloc[i:i+7]
        #     all_windows.append(window)
    
    # stacked_df = pd.concat(all_windows, keys=range(len(all_windows)))
    # return stacked_df


def plot_distribution(df, column_name, stock_name):
    """
    Plots the probabilistic distribution of a specified column in a DataFrame.

    :param df: Pandas DataFrame containing the data.
    :param column_name: Name of the column to plot. Defaults to 'Diff'.
    """

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' is not a column in the DataFrame.")

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    sns.kdeplot(df[column_name], fill=True, color='blue')

    # Plot formatting
    plt.grid(True)
    plt.title('Probabilistic Distribution of Close-Close Difference for stock {}'.format(stock_name))
    plt.xlabel(column_name)
    plt.ylabel('Density')
    
    # Show the plot
    plt.show()
