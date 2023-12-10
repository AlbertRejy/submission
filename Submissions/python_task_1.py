import pandas as pd


def generate_car_matrix(df):
    # Pivot the DataFrame to get id_1 as index, id_2 as columns, and car as values
    pivot_df = df.pivot(index='id_1', columns='id_2', values='car')
    
    # Fill NaN values with 0
    pivot_df = pivot_df.fillna(0)
    
    # Set diagonal values to 0
    for col in pivot_df.columns:
        pivot_df.at[col, col] = 0
    
    return pivot_df


# Read the dataset-1.csv file into a DataFrame
df = pd.read_csv('dataset-1.csv')

# Call the function with the DataFrame
result_matrix = generate_car_matrix(df)

# Display the result
print(result_matrix)





df.head()

def get_type_count(df):
    # Add a new categorical column 'car_type' based on the values of the 'car' column
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'])
    
    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()
    
    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))
    
    return sorted_type_counts



# Call the function with the DataFrame
result = get_type_count(df)

# Display the result
print(result)

'''


'''

def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


# Call the function with the DataFrame
result = get_bus_indexes(df)

# Display the result
print(result)






def filter_routes(df):
    # Group by 'route' and calculate the average of 'truck' for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of routes in ascending order
    filtered_routes.sort()

    return filtered_routes



# Call the function with the DataFrame
result = filter_routes(df)

# Display the result
print(result)


def multiply_matrix(result_matrix):
    # Create a deep copy of the DataFrame to avoid modifying the original
    modified_matrix = result_matrix.copy()

    # Apply the multiplication logic
    modified_matrix[modified_matrix > 20] *= 0.75
    modified_matrix[modified_matrix <= 20] *= 1.25

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Assuming result_matrix is the DataFrame from Question 1
# Call the function with the DataFrame
modified_result_matrix = multiply_matrix(result_matrix)

# Display the modified DataFrame
print(modified_result_matrix)




from dateutil.relativedelta import relativedelta

def verify_timestamp_completeness(df):
    # Convert timestamp columns to datetime objects
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Create a 24-hour time range for each day
    full_24_hours = pd.date_range('00:00:00', '23:59:59', freq='1S').time

    # Create a 7-day week
    full_week = pd.date_range('00:00:00', '23:59:59', freq='1D').day_name()

    # Check if each timestamp covers a full 24-hour period and spans all 7 days
    completeness_check = (
        df.groupby(['id', 'id_2'])
        .apply(lambda group: all(group['start_timestamp'].dt.time.isin(full_24_hours)) and
                             all(group['end_timestamp'].dt.time.isin(full_24_hours)) and
                             set(group['start_timestamp'].dt.day_name()) == set(full_week) and
                             set(group['end_timestamp'].dt.day_name()) == set(full_week))
    )

    return completeness_check

# Read the dataset-2.csv file into a DataFrame
df_dataset_2 = pd.read_csv('dataset-2.csv')

# Call the function with the DataFrame
completeness_result = verify_timestamp_completeness(df_dataset_2)

# Display the result
print(completeness_result)
