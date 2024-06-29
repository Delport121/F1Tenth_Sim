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

def plot_data(data):
    x_values = [row[0] for row in data]
    y_values = [row[1] for row in data]

    plt.scatter(x_values, y_values, label='Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of the Given Set of Points')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(script_directory, "gbr_centerline.csv")
    
    try:
        data = read_csv(csv_file_path)
        plot_data(data)
    except Exception as e:
        print(f"Error: {e}")
