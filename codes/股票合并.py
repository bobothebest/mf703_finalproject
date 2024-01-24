#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:51:58 2023

@author: apple
"""
import pandas as pd

# Function to load and merge two Excel files without changing the first row and remove duplicates
def merge_excel_files_without_duplicates(file_path1, file_path2, output_file_path):
    # Read the files into pandas DataFrames
    df1 = pd.read_excel(file_path1)
    df2 = pd.read_excel(file_path2)


    df1 = df1.rename(columns={df1.columns[0]: 'Date'})
    df2 = df2.rename(columns={df2.columns[0]: 'Date'})

    # Drop the first column of df2
    df2 = df2.drop(columns=df2.columns[0])
    # 合并 df1 和 df2
    
    merged_df = pd.concat([df1, df2], axis=1)

    # Truncate each column name to the first 9 characters
    merged_df.columns = [col[:9] for col in merged_df.columns]
    
    duplicates = merged_df.columns.duplicated()
    
    merged_df = merged_df.loc[:, ~duplicates]
    
    merged_df.columns = [col[:9] for col in merged_df.columns]
    
    # Save the merged dataframe to a new Excel file
    merged_df.to_excel(output_file_path, index=False)
    
    return output_file_path

# Specify the paths to the two Excel files and the output file
file_path1 = '/Users/apple/Downloads/科科/MF703_FinalProject/data/stock.xlsx'
file_path2 = '/Users/apple/Downloads/科科/MF703_FinalProject/data/stock_extra.xlsx'
output_file_path = '/Users/apple/Downloads/科科/MF703_FinalProject/data/stock_full.xlsx'


# Call the function to merge the Excel files without duplicate columns
merged_file_path_no_duplicates = merge_excel_files_without_duplicates(file_path1, file_path2, output_file_path)
print(f"Merged Excel file without duplicates saved to: {merged_file_path_no_duplicates}")

