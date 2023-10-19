
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
    
    data['Close_Diff_3_days'] = data['Close'].shift(-2) - data['Close']
    data['Close_Diff_5_days'] = data['Close'].shift(-4) - data['Close']
    data['Close_Diff_7_days'] = data['Close'].shift(-6) - data['Close']

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

        dataset['Close_Diff_3_days'] = dataset['Close'].shift(-2) - dataset['Close']
        dataset['Close_Diff_5_days'] = dataset['Close'].shift(-4) - dataset['Close']
        dataset['Close_Diff_7_days'] = dataset['Close'].shift(-6) - dataset['Close']

        dataset = dataset.sort_index()
        dataset['Quantile_33_3_days'] = dataset['Close_Diff_3_days'].quantile(0.33)
        dataset['Quantile_66_3_days'] = dataset['Close_Diff_3_days'].quantile(0.66)
        dataset['Quantile_33_5_days'] = dataset['Close_Diff_5_days'].quantile(0.33)
        dataset['Quantile_66_5_days'] = dataset['Close_Diff_5_days'].quantile(0.66)
        dataset['Quantile_33_7_days'] = dataset['Close_Diff_7_days'].quantile(0.33)
        dataset['Quantile_66_7_days'] = dataset['Close_Diff_7_days'].quantile(0.66)

        if stock_name in tech_companies:
            quantile_values = [dataset['Quantile_33_3_days'].iloc[0], 
                               dataset['Quantile_66_3_days'].iloc[0],
                               dataset['Quantile_33_5_days'].iloc[0],
                               dataset['Quantile_66_5_days'].iloc[0],
                               dataset['Quantile_33_7_days'].iloc[0],
                               dataset['Quantile_66_7_days'].iloc[0]]
                               
            plot_distribution(dataset, 'Close_Diff_3_days', quantile_values[0], quantile_values[1], stock_name)
            plot_distribution(dataset, 'Close_Diff_5_days', quantile_values[2], quantile_values[3], stock_name)
            plot_distribution(dataset, 'Close_Diff_7_days', quantile_values[4], quantile_values[5], stock_name)

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

        datasets.append(dataset)
        
    return datasets


def plot_distribution(df, column_name, quantile_33, quantile_66, stock_name):
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

    data = df[column_name].dropna()

    # Creating the histogram. The 'density=True' parameter ensures that we create a probability density histogram
    # where the sum of all bins' heights equals 1, representing the total probability.
    plt.hist(data, bins=500, density=True, alpha=0.75, color='blue')  # 'bins' can be adjusted depending on your data

    # Quantile line plot
    plt.axvline(x=quantile_33, color='red', linestyle='--', linewidth=2, label=f'Quantile_33: {quantile_33}')
    plt.axvline(x=quantile_66, color='red', linestyle='--', linewidth=2, label=f'Quantile_66: {quantile_66}')

    ticks = [tick for tick in plt.xticks()[0] if tick != 0.0]  # get the current tick locations and labels

    # Adding the new tick for the quantile value (rounding and formatting for better appearance)
    new_ticks = [round(quantile_33, 3), round(quantile_66, 3)]  # or use another criterion suitable for your values
    ticks = list(ticks) + new_ticks

    # Set the ticks to the x-axis, this includes the original and the new tick for the quantile value
    plt.xticks(ticks, [str(tick) for tick in ticks])  # converting values to string for proper display


    # Plot formatting
    plt.grid(True)
    plt.title('Probabilistic Distribution of Close-Close Difference for stock {}'.format(stock_name))
    plt.xlabel(column_name)
    plt.ylabel('Density')

    plt.xticks(rotation=90)
    
    # Show the plot
    plt.show()
