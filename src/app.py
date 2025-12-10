from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# Goal: give a clear and summarized description of the data in the New York housing rental data from Airbnb 2019

# Step 1: Download New York Airbnb data from Kaggle.com.
main_df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv") 

# Step 2: All changes made to the dataframe during EDA

drop_list = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'latitude', 'longitude', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
main_df.drop(drop_list, axis = 1, inplace = True)
main_df = main_df[main_df["price"] != 0]

room_codes,room_unique = pd.factorize(main_df["room_type"], sort=True)
neighbourhood_codes,neighbourhood_unique = pd.factorize(main_df["neighbourhood_group"], sort=True)
main_df["room_type"] = room_codes
main_df["neighbourhood_group"] = neighbourhood_codes

from sklearn.preprocessing import MinMaxScaler

num_variables = ["number_of_reviews", "minimum_nights","neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(main_df[num_variables])
df_scal = pd.DataFrame(scal_features, index = main_df.index, columns = num_variables)
df_scal["price"] = main_df["price"]

# Step 3: Write down the conclusions of each step and analyze the results on the relationships between the variables.
# I dropped columns from the dataframe that were a) names/identifiers b) were related to other columns and overly specific or irrelevant c) had too many null values 
# I then dropped 0 values from the price column as they were likely to be errors or outliers
# I factorized the categorical columns to convert them to numerical values for easier analysis
# I normalized the numerical columns using MinMaxScaler 
# Finally I split and saved the cleaned dataframe as a new csv file