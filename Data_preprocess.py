import pandas as pd

# Load the data from the provided text file
data = pd.read_csv('extracted_data.csv')

# Display the first few rows of the original data
print("Original Data:")
print(data.head())

# Function to remove HTML tags and handle 'Not Available' entries
def clean_text(column):
    # Remove HTML tags
    column = column.str.replace('<[^<]+?>', '', regex=True)
    # Replace 'Not Available' with an empty string
    column = column.replace('Not Available', '')
    return column

# Apply cleaning function to all columns except 'url'
for col in data.columns:
    if col != 'url':
        data[col] = clean_text(data[col])

# Display the cleaned data
print("\nCleaned Data:")
print(data.head())

# Save the cleaned data to a new CSV file
data.to_csv('cleaned_data.csv', index=False)
