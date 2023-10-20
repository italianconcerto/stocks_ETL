
import os
from tqdm import tqdm
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

# top5 = ["BIIB", "ORLY", "VRTX", "AMGN", "COST"]
top5 = ["NVDA", "CMCSA", "FAST", "CSCO", "EBAY", "ATVI"]
closest_mean_median = ["ADP", "PCAR"]

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

        if stock_name in closest_mean_median:
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

        datasets.append((dataset, stock_name))
        
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


def extract_top_diff_datasets(all_data, far=True):
    # Creating a dictionary to store the results
    results = {
        '3_days': [],
        '5_days': [],
        '7_days': []
    }

    # Function to get top 5 datasets based on quantile differences
    def get_top_5(all_data, quantile_33_col, quantile_66_col, far):
        quantile_diffs = []
        for data in all_data:
            # Calculate the difference between the 66th and 33rd quantiles
            dataset = data[0]
            stock_name = data[1]
            quantile_diff = dataset[quantile_66_col].iloc[0] - dataset[quantile_33_col].iloc[0]
            quantile_diffs.append((stock_name, dataset, quantile_diff))

        # Sort the datasets based on the quantile difference
        quantile_diffs.sort(key=lambda x: x[2], reverse=far)

        # Return the top 5 datasets
        return [(item[1], item[0]) for item in quantile_diffs[:5]]
    
    import pandas as pd  # make sure pandas is imported

    def compute_closest(all_data, quantile_33_col, quantile_66_col, metric):
        # Define the central tendency function based on the metric
        central_tendency = {
            'mean': pd.Series.mean,
            'median': pd.Series.median
        }.get(metric)

        if not central_tendency or not all_data:
            return None  # Return None for invalid metric or empty data

        quantile_diffs = pd.Series([data[0][quantile_66_col].iloc[0] - data[0][quantile_33_col].iloc[0] for data in all_data])
        central_value = central_tendency(quantile_diffs)

        # Find and return the dataset closest to the central tendency value
        closest_data = min(all_data, key=lambda x: abs((x[0][quantile_66_col].iloc[0] - x[0][quantile_33_col].iloc[0]) - central_value), default=None)
        
        # The closest_data contains the dataset and the stock name, you can return as needed
        return (closest_data, central_value)  # or closest_data[1] for stock_name only or closest_data[0] for dataset only


    # Extract top 5 for 3 days, 5 days, and 7 days
    results['top5_3_days'] = get_top_5(all_data, 'Quantile_33_3_days', 'Quantile_66_3_days', far)
    results['top5_5_days'] = get_top_5(all_data, 'Quantile_33_5_days', 'Quantile_66_5_days', far)
    results['top5_7_days'] = get_top_5(all_data, 'Quantile_33_7_days', 'Quantile_66_7_days', far)

    results['closest_mean_3_days'] = compute_closest(all_data, 'Quantile_33_3_days', 'Quantile_66_3_days', 'mean')
    results['closest_median_3_days'] = compute_closest(all_data, 'Quantile_33_3_days', 'Quantile_66_3_days', 'median')

    results['closest_mean_5_days'] = compute_closest(all_data, 'Quantile_33_5_days', 'Quantile_66_5_days', 'mean')
    results['closest_median_5_days'] = compute_closest(all_data, 'Quantile_33_5_days', 'Quantile_66_5_days', 'median')

    results['closest_mean_7_days'] = compute_closest(all_data, 'Quantile_33_7_days', 'Quantile_66_7_days', 'mean')
    results['closest_median_7_days'] = compute_closest(all_data, 'Quantile_33_7_days', 'Quantile_66_7_days', 'median')

    return results

def print_info(data):
    print("\n" + "----- MAX DIFF -----" + "\n")
    top_datasets = extract_top_diff_datasets(data)
    print("MAX Q33-Q66 diff 3 days: \n")
    [print(top_datasets['top5_3_days'][i][1] + ", Quantile33: " + str(top_datasets['top5_3_days'][i][0]['Quantile_33_3_days'][0]) + ", Quantile66: " + str(top_datasets['top5_3_days'][i][0]['Quantile_66_3_days'][0])) for i in range(0,5)]
    print("MAX Q33-Q66 diff 5 days: \n")
    [print(top_datasets['top5_5_days'][i][1] + ", Quantile33: " + str(top_datasets['top5_5_days'][i][0]['Quantile_33_5_days'][0]) + ", Quantile66: " + str(top_datasets['top5_5_days'][i][0]['Quantile_66_5_days'][0])) for i in range(0,5)]
    print("MAX Q33-Q66 diff 7 days: \n")
    [print(top_datasets['top5_7_days'][i][1] + ", Quantile33: " + str(top_datasets['top5_7_days'][i][0]['Quantile_33_7_days'][0]) + ", Quantile66: " + str(top_datasets['top5_7_days'][i][0]['Quantile_66_7_days'][0])) for i in range(0,5)]

    print("\n" + "----- MIN DIFF -----" + "\n")
    top_datasets = extract_top_diff_datasets(data, False)
    print("MIN Q33-Q66 diff 3 days: \n")
    [print(top_datasets['top5_3_days'][i][1] + ", Quantile33: " + str(top_datasets['top5_3_days'][i][0]['Quantile_33_3_days'][0]) + ", Quantile66: " + str(top_datasets['top5_3_days'][i][0]['Quantile_66_3_days'][0])) for i in range(0,5)]
    print("MIN Q33-Q66 diff 5 days: \n")
    [print(top_datasets['top5_5_days'][i][1] + ", Quantile33: " + str(top_datasets['top5_5_days'][i][0]['Quantile_33_5_days'][0]) + ", Quantile66: " + str(top_datasets['top5_5_days'][i][0]['Quantile_66_5_days'][0])) for i in range(0,5)]
    print("MIN Q33-Q66 diff 7 days: \n")
    [print(top_datasets['top5_7_days'][i][1] + ", Quantile33: " + str(top_datasets['top5_7_days'][i][0]['Quantile_33_7_days'][0]) + ", Quantile66: " + str(top_datasets['top5_7_days'][i][0]['Quantile_66_7_days'][0])) for i in range(0,5)]



    print("\n" + "----- CLOSEST MEAN -----" + "\n")
    print("Quantile diff mean 3 days: " + str(top_datasets['closest_mean_3_days'][1]))
    print("Quantile diff mean 5 days: " + str(top_datasets['closest_mean_5_days'][1]))
    print("Quantile diff mean 7 days: " + str(top_datasets['closest_mean_7_days'][1]))
    dataset, stock_name = top_datasets['closest_mean_3_days'][0]
    print("\nClosest to mean: " + stock_name + ", Quantile33: " + str(top_datasets['closest_mean_3_days'][0][0]['Quantile_33_3_days'][0]) + ", Quantile66: " + str(top_datasets['closest_mean_3_days'][0][0]['Quantile_66_3_days'][0]))
    dataset, stock_name = top_datasets['closest_mean_5_days'][0]
    print("\nClosest to mean: " + stock_name + ", Quantile33: " + str(top_datasets['closest_mean_5_days'][0][0]['Quantile_33_5_days'][0]) + ", Quantile66: " + str(top_datasets['closest_mean_5_days'][0][0]['Quantile_66_5_days'][0]))
    dataset, stock_name = top_datasets['closest_mean_7_days'][0]
    print("\nClosest to mean: " + stock_name + ", Quantile33: " + str(top_datasets['closest_mean_7_days'][0][0]['Quantile_33_7_days'][0]) + ", Quantile66: " + str(top_datasets['closest_mean_7_days'][0][0]['Quantile_66_7_days'][0]))

    print("\n" + "----- CLOSEST MEDIAN -----" + "\n")
    print("Quantile diff median 3 days: " + str(top_datasets['closest_median_3_days'][1]))
    print("Quantile diff median 5 days: " + str(top_datasets['closest_median_5_days'][1]))
    print("Quantile diff median 7 days: " + str(top_datasets['closest_median_7_days'][1]))
    dataset, stock_name = top_datasets['closest_median_3_days'][0]
    print("\nClosest to median: " + stock_name + ", Quantile33: " + str(top_datasets['closest_median_3_days'][0][0]['Quantile_33_3_days'][0]) + ", Quantile66: " + str(top_datasets['closest_median_3_days'][0][0]['Quantile_66_3_days'][0]))
    dataset, stock_name = top_datasets['closest_median_5_days'][0]
    print("\nClosest to median: " + stock_name + ", Quantile33: " + str(top_datasets['closest_median_5_days'][0][0]['Quantile_33_5_days'][0]) + ", Quantile66: " + str(top_datasets['closest_median_5_days'][0][0]['Quantile_66_5_days'][0]))
    dataset, stock_name = top_datasets['closest_median_7_days'][0]
    print("\nClosest to median: " + stock_name + ", Quantile33: " + str(top_datasets['closest_median_7_days'][0][0]['Quantile_33_7_days'][0]) + ", Quantile66: " + str(top_datasets['closest_median_7_days'][0][0]['Quantile_66_7_days'][0]))