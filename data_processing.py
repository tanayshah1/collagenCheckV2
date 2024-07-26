import numpy as np

def clean_data(input_filename, output_filename):
    cleaned_lines = []

    # Read the input file and remove the first value from each line
    with open(input_filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            values = list(map(float, line.strip().split()))
            if len(values) > 1:
                # Remove the first value
                values.pop(0)
                cleaned_lines.append((line_number, values))
    
    # Write the cleaned data to the output CSV file
    with open(output_filename, 'w') as output_file:
        # Write header to CSV file (optional)
        #output_file.write('Line Number,Values\n')
        
        for line_number, values in cleaned_lines:
            # Calculate the median of the remaining values
            if values:
                median = np.median(values)
                
                # Check if any value is at least 100 above the median
                if any(value >= median + 100 for value in values):
                    # Write the line number and the entire cleaned line to the output file
                    output_file.write(f"{line_number} {' '.join(map(str, values))}\n")
    
    # Print the number of lines processed
    print(f"Total lines processed and saved to {output_filename}")

# Process control data
clean_data('/Users/tanayshah/vscode/collegenFiberCheck/Graph Data/May 16 Control C-H.txt', 'cleaned_control_data.csv')

# Process test data
clean_data('/Users/tanayshah/vscode/collegenFiberCheck/Graph Data/May 16 Tumor C-H.txt', 'cleaned_test_data.csv')
