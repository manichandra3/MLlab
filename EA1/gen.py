import numpy as np
import pandas as pd

# Define the number of samples
num_samples = 10000

# Generate random data
np.random.seed(0)  # For reproducibility

# Features
sizes = np.random.randint(1000, 4000, size=num_samples)
bedrooms = np.random.randint(1, 6, size=num_samples)  # Number of bedrooms
ages = np.random.randint(0, 100, size=num_samples)  # Age in years
distances = np.random.uniform(1, 15, size=num_samples)

# Prices - generate a somewhat realistic price based on the features
prices = sizes * 150 + bedrooms * 5000 - ages * 500 + distances * 20000 + np.random.normal(0, 10000, num_samples)

# Create a DataFrame
data = {
    'Size (sq ft)': sizes,
    'Bedrooms': bedrooms,
    'Age (years)': ages,
    'Distance to City Center (miles)': distances,
    'Price ($)': prices
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('large_house_prices_data.csv', index=False)

print('CSV file with random data has been created.')
