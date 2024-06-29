import csv
import os
import matplotlib.pyplot as plt

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # Skip the header if present
        next(reader, None)
        data = [list(map(float, row)) for row in reader]
    return data

def plot_columns(data):
    x_values = [row[2] for row in data]  # Third column as x-coordinates
    y_values = [row[3] for row in data]  # Fourth column as y-coordinates

    plt.scatter(x_values, y_values, label='Points')
    plt.xlabel('Column 3')
    plt.ylabel('Column 4')
    plt.title('Scatter Plot of Column 3 vs Column 4')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(script_directory, "gbr_centerline.csv")
    
    try:
        data = read_csv(csv_file_path)
        plot_columns(data)
    except Exception as e:
        print(f"Error: {e}")
