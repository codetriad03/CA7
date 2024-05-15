import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
with open('news_headlines.txt', 'r') as file:
    headlines = file.readlines()

# Preprocess data
headlines = [headline.strip() for headline in headlines]
headlines, sources = zip(*[headline.split(' - ') for headline in headlines])

# Split data into training, validation, and testing sets
train_data, test_data = train_test_split(list(zip(headlines, sources)), test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Create the 'data' directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Save the datasets
train_df = pd.DataFrame(train_data, columns=['headline', 'source'])
val_df = pd.DataFrame(val_data, columns=['headline', 'source'])
test_df = pd.DataFrame(test_data, columns=['headline', 'source'])

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/validation.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
