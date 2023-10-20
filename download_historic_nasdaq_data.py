import glob
from tqdm import tqdm
import pandas as pd
from config import *
from utils import delete_files_in_folders, get_stock_data, process_datasets, print_info
import os

indices = nasdaq_100_companies #tech_companies + healthcare_companies + finance_companies + manufacturing_companies

if CLEAR_STOCKS_FILES:
    delete_files_in_folders(stocks_folder)
    for index in indices:
        print('Downloading', index)
        data = get_stock_data(index, start=start, end=end, interval='1d')
        if data is None:
            print('Data not consistent for ', index)
            continue
        data.to_csv('data/stocks/nasdaq_100/' + index + '.csv')
        print('Downloaded', index)
        
    # for index in healthcare_companies:
    #     print('Downloading', index)
    #     data = get_stock_data(index, start=start, end=end, interval='1d')
    #     if data is None:
    #         print('Data not consistent for ', index)
    #         continue
    #     data.to_csv('data/stocks/healthcare/' + index + '.csv')
    #     print('Downloaded', index)
        
        
    # for index in finance_companies:
    #     print('Downloading', index)
    #     data = get_stock_data(index, start=start, end=end, interval='1d')
    #     if data is None:
    #         print('Data not consistent for ', index)
    #         continue
    #     data.to_csv('data/stocks/finance/' + index + '.csv')
    #     print('Downloaded', index)
        

    # for index in manufacturing_companies:
    #     print('Downloading', index)
    #     data = get_stock_data(index, start=start, end=end, interval='1d')
    #     if data is None:
    #         print('Data not consistent for ', index)
    #         continue
    #     data.to_csv('data/stocks/manufacturing/' + index + '.csv')
    #     print('Downloaded', index)
        
    
# Create a list of dataframe by reading all the csv files in data/stocks folder using glob.glob
stocks = []

print("Reading csv files...")
for file in tqdm(glob.glob(stocks_folder + "/*/*.csv")):
    df = pd.read_csv(file)
    stock_name = os.path.basename(file).replace('.csv', '')
    stocks.append((df, stock_name))

print("Processing datasets...")
data = process_datasets(stocks)
print_info(data)


breakpoint()