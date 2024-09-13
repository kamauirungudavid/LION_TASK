import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns



def load_data(data_path):
    """
    Helps loading data. tries csv and excel
    INPUT:
    ---------
    data_path
    ----------

    OUTPUT
    ---------
    - raw_data
    ---------
    """
    try:
        data = pd.read_csv(f'{data_path}')
        logging.error('SUCCESS LOADING DATA')
    except Exception as e:
        logging.error(f"error reading csv_file:{e}")

    try: 
        data = pd.read_excel(f'{data_path}')
        logging('SUCCESS LOADING DATA')
    except Exception as e:
        logging.error(f"error reading excel_file:{e}")
    return data

def data_cleaning(data):
    """
    Conduct Data Cleaning

    INPUT:
    ---------
    raw_data
    ----------

    OUTPUT
    ---------
    - clean data
    ---------
    """
    # missing values
    try:
        mis_val = data.isnull().sum()
        mis_val_percent = 100 * data.isnull().sum() / len(data)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1).reset_index()
        mis_val_table.columns = ['variable','missing','perc_missing']
        cols_to_remove = []
        thresh_hold = 90
        for index, row in mis_val_table.iterrows():
            if row['perc_missing'] >= thresh_hold:
                cols_to_remove.append(row['variable'])

    except Exception as e: 
        logging.error(f'error defining columns to remove::{e}')

        #can not impute target(drop those with missing)
    if len(cols_to_remove) != 0:
        data.drop(columns=cols_to_remove, inplace=True)


    return data


def EDA(data, wkdir):
    """
    Conduct Exploratory Data Analytics

    INPUT:
    ---------
    clean_data
    ----------

    OUTPUT
    ---------
    - visualizations and additional features
    - clean data
    ---------
    """
    # visualize target distribution
    try:
        plt.figure(figsize=(13, 4))

        # Pie chart
        plt.subplot(121)
        data["Default"].value_counts().plot.pie(
            autopct="%1.0f%%", colors=sns.color_palette("prism", 7),
            startangle=60, labels=["good", "bad"],
            wedgeprops={"linewidth": 2, "edgecolor": "k"}, explode=[.1, 0], shadow=True
        )
        plt.title("Distribution of Target Variable")

        # Bar chart
        plt.subplot(122)
        ax = data["Default"].value_counts().plot(kind="barh")
        for i, j in enumerate(data["Default"].value_counts().values):
            ax.text(.7, i, j, weight="bold", fontsize=20)
        plt.title("Count of Target Variable")
        plt.savefig(f"{wkdir}/outputs/TARGET_DISTRIB.png")
    except Exception as e:
        logging.error(f'Could not log target distirbution::{e}')
 
    #checking quasi constants
    try:
        constant_cols = [c for c in data.columns if len(data[c].unique()) == 1]
        logging(f"{len(constant_cols)} columns with a unique value in the dataset.")
    except Exception as e:
        logging.error(f'could not check constant columns::{e}')

    try:
        # generic customer_id (txn_id+name)
        data['customer_id'] = (data['Customer Name']) + \
            data['Transaction ID'].astype(str)

        # feature engineering
        data['income_balance_ratio'] = data['Income']/data['Account Balance']

        # imput Amount
        data['Amount'] = data.groupby(['Class of Business', 'Transaction Type', 'Default'])[
            'Amount'].transform(lambda x: x.fillna(x.mean()))

        # change currency to KES/USD
        current_fx = 130  # assumption here
        data['Amount_converted'] = np.where(data['Currency'] == 'USD',
                                            data['Amount'] * current_fx,
                                            data['Amount'])

        # checking outliers and spacial values
        data['Date'] = pd.to_datetime(data['Date'])
        data['inactive_days'] = (pd.Timestamp(
            'today') - data['Date']).dt.days

        #checking multicolinearity
    except Exception as e:
        logging.error(f'Error in EDA:: {e}')

    #
    return data


def get_IRA_format(data,index1_col, index2_col, value_cols,wdir):
    try:
        cross_tab = pd.crosstab([data[index1_col], data[index2_col]],
                                data[value_cols], margins=True, margins_name='Total')
        cross_tab = cross_tab.reset_index()
        cross_tab.to_csv(
            f'{wdir}/outputs/IRA_data.csv', index=False)
        return cross_tab
    except Exception as e:
        logging.error(f'Error generating IRA formated data :: {e}')

def main():
    wdir = os.getcwd()
    data_path = f'{wdir}/data/ILTESTData_Raw_20240909.xlsx'
    loaded_data = load_data(data_path)
    clean_data = data_cleaning(loaded_data)
    clean_data2 = EDA(clean_data, wdir)
    print(clean_data.head())

    IRA_data = get_IRA_format(data=clean_data2, index1_col='Class of Business', index2_col='Sub-Class of Business',
    value_cols = 'Transaction Type', wdir = wdir)


    #save clean data for modeling
    clean_data2.to_csv(f'{wdir}/outputs/data_for_modeling.csv', index=False)



if __name__ == "__main__":
    main()






