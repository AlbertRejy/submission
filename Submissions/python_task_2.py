import pandas as pd

def calculate_distance_matrix(df):
    # Create a DataFrame with unique IDs as both index and columns
    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)

    # Initialize the distance matrix with zeros
    distance_matrix = distance_matrix.fillna(0)

    # Update the matrix with cumulative distances
    for index, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']

        # Update the distance between id_start and id_end
        distance_matrix.at[id_start, id_end] += distance

        # Ensure symmetry by updating the distance between id_end and id_start
        distance_matrix.at[id_end, id_start] += distance

    return distance_matrix

# Read the dataset-3.csv file into a DataFrame
df_dataset_3 = pd.read_csv('dataset-3.csv')

# Call the function with the DataFrame
distance_matrix = calculate_distance_matrix(df_dataset_3)

# Display the resulting distance matrix
print(distance_matrix)






import numpy as np

def unroll_distance_matrix(distance_matrix):
    # Ensure the input matrix is symmetric and has valid index and columns
    assert (distance_matrix.index == distance_matrix.columns).all(), "Distance matrix must be symmetric"
    
    # Get the upper triangular part of the matrix (excluding the diagonal)
    upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1) == 1)
    
    # Reset the index to get 'id_start' and 'id_end' as columns
    unrolled_df = upper_triangle.stack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    
    return unrolled_df

# Assuming distance_matrix is the DataFrame from Question 1
# Call the function with the DataFrame
unrolled_df = unroll_distance_matrix(distance_matrix)

# Display the resulting unrolled DataFrame
print(unrolled_df)




def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Filter the DataFrame for rows with the given reference value
    reference_rows = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_average_distance = reference_rows['distance'].mean()

    # Calculate the threshold range (within 10% of the reference average)
    lower_threshold = reference_average_distance - 0.1 * reference_average_distance
    upper_threshold = reference_average_distance + 0.1 * reference_average_distance

    # Filter the DataFrame for rows within the threshold range and get unique values
    filtered_rows = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]
    unique_ids_within_threshold = sorted(filtered_rows['id_start'].unique())

    return unique_ids_within_threshold

# Assuming unrolled_df is the DataFrame from the previous step
# Reference value for demonstration purposes
reference_value = 100

# Call the function with the DataFrame and reference value
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)

# Display the sorted list of values within the 10% threshold
print(result_ids)





def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type with toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

# Assuming unrolled_df is the DataFrame from the previous step
# Call the function with the DataFrame
df_with_toll_rates = calculate_toll_rate(unrolled_df)

# Display the resulting DataFrame with toll rates
print(df_with_toll_rates)










# Assuming you have a DataFrame named 'df' created in Question 3

def calculate_time_based_toll_rates(dataframe):
    # Define time ranges and discount factors
    weekday_discounts = [(time(0, 0, 0), time(10, 0, 0), 0.8),
                         (time(10, 0, 0), time(18, 0, 0), 1.2),
                         (time(18, 0, 0), time(23, 59, 59), 0.8)]

    weekend_discount = 0.7

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate over unique ('id', 'name', 'id_2') triplets
    for (id_val, name_val, id_2_val), group_df in dataframe.groupby(['id', 'name', 'id_2']):
        for day_offset in range(7):
            for start_time, end_time, discount_factor in weekday_discounts:
                start_datetime = datetime.combine(datetime.today(), start_time) + timedelta(days=day_offset)
                end_datetime = datetime.combine(datetime.today(), end_time) + timedelta(days=day_offset)
                
                # Apply discount factor based on the time range
                group_df.loc[(group_df['timestamp'] >= start_datetime) & (group_df['timestamp'] <= end_datetime),
                             ['able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3', 'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3']] *= discount_factor

                # Create a row for each time range
                row = {
                    'id': id_val,
                    'name': name_val,
                    'id_2': id_2_val,
                    'startDay': (datetime.today() + timedelta(days=day_offset)).strftime('%A'),
                    'endDay': (datetime.today() + timedelta(days=day_offset)).strftime('%A'),
                    'startTime': start_time,
                    'endTime': end_time
                }
                result_df = result_df.append(row, ignore_index=True)

        # Apply constant discount factor for weekends
        for day_offset in range(5, 7):
            group_df.loc[:, ['able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3', 'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3']] *= weekend_discount

            # Create a row for the weekend time range
            row = {
                'id': id_val,
                'name': name_val,
                'id_2': id_2_val,
                'startDay': (datetime.today() + timedelta(days=day_offset)).strftime('%A'),
                'endDay': (datetime.today() + timedelta(days=day_offset)).strftime('%A'),
                'startTime': time(0, 0, 0),
                'endTime': time(23, 59, 59)
            }
            result_df = result_df.append(row, ignore_index=True)

    # Merge the original DataFrame with the calculated rates
    result_df = pd.merge(dataframe, result_df, on=['id', 'name', 'id_2', 'startDay', 'endDay', 'startTime', 'endTime'])

    return result_df

# Example usage:
# result_df = calculate_time_based_toll_rates(df)







