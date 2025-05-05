import pandas as pd

def convert_file(input_file, output_file, delimiter=None):
    try:
        # Auto-detect delimiter if not specified
        if delimiter is None:
            with open(input_file, 'r') as f:
                first_line = f.readline()
            delimiter = '\t' if '\t' in first_line else ',' if ',' in first_line else ' '
        
        # Read file with flexible options
        df = pd.read_csv(
            input_file,
            delimiter=delimiter,
            engine='python',
            header=None if 'your_header' not in first_line else 'infer',
            skipinitialspace=True,
            error_bad_lines=False
        )
        
        # Save as CSV
        df.to_csv(output_file, index=False)
        print(f"✅ Success! Saved as {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTry specifying the delimiter:")
        print("convert_file('input.txt', 'output.csv', delimiter='\\t')")

# Example usage (modify as needed):
convert_file('clinical_data.txt', 'clinical_data.csv')