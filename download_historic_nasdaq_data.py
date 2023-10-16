

import glob

import pandas as pd
from config import *
from utils import delete_files_in_folders, get_stock_data, process_datasets



indices = healthcare_indices + tech_companies + finance_companies + manufacturing_companies






if CLEAR_STOCKS_FILES:
    delete_files_in_folders(stocks_folder)


if CLEAR_STOCKS_FILES:
    for index in tech_companies:
        print('Downloading', index)
        data = get_stock_data(index, start=start, end=end, interval='1d')
        if data is None:
            print('Data not consistent for ', index)
            continue
        data.to_csv('data/stocks/tech/' + index + '.csv')
        print('Downloaded', index)
        
    for index in healthcare_indices:
        print('Downloading', index)
        data = get_stock_data(index, start=start, end=end, interval='1d')
        if data is None:
            print('Data not consistent for ', index)
            continue
        data.to_csv('data/stocks/healthcare/' + index + '.csv')
        print('Downloaded', index)
        
        
    for index in finance_companies:
        print('Downloading', index)
        data = get_stock_data(index, start=start, end=end, interval='1d')
        if data is None:
            print('Data not consistent for ', index)
            continue
        data.to_csv('data/stocks/finance/' + index + '.csv')
        print('Downloaded', index)
        

    for index in manufacturing_companies:
        print('Downloading', index)
        data = get_stock_data(index, start=start, end=end, interval='1d')
        if data is None:
            print('Data not consistent for ', index)
            continue
        data.to_csv('data/stocks/manufacturing/' + index + '.csv')
        print('Downloaded', index)
        
    
# Create a list of dataframe by reading all the csv files in data/stocks folder using glob.glob
stocks = []


from tqdm import tqdm

print("Reading csv files...")
for file in tqdm(glob.glob(stocks_folder + "/*/*.csv")):
    df = pd.read_csv(file)
    stocks.append(df)



print("Processing datasets...")
stacked_data = process_datasets(stocks)
breakpoint()
