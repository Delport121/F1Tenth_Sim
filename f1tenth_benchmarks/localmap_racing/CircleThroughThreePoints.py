import math
import matplotlib.pyplot as plt

def calculate_circle_radius_and_center(x1, y1, x2, y2, x3, y3):
    # Calculate the determinant (related to twice the area of the triangle)
    A = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    
    if A == 0:
        raise ValueError("The points are collinear, so a unique circle cannot be formed.")

    # Calculate the squared lengths of the sides of the triangle
    a_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    b_sq = (x3 - x2) ** 2 + (y3 - y2) ** 2
    c_sq = (x1 - x3) ** 2 + (y1 - y3) ** 2

    # Calculate the circumradius using the correct formula
    a = math.sqrt(a_sq)
    b = math.sqrt(b_sq)
    c = math.sqrt(c_sq)
    
    # Area of the triangle (using determinant method)
    area = abs(A) / 2

    # Circumradius
    radius = (a * b * c) / (4 * area)

    # Calculate the circumcenter coordinates
    x_center = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / (2 * A)
    y_center = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / (2 * A)

    return radius, (x_center, y_center)

def plot_circle(radius, center, points):
    fig, ax = plt.subplots()
    
    # Circle
    circle = plt.Circle(center, radius, color='blue', fill=False, linestyle='--')
    ax.add_artist(circle)
    
    # Plot the three points
    for point in points:
        ax.plot(point[0], point[1], 'ro')
    
    # Plot the center of the circle
    ax.plot(center[0], center[1], 'bo')
    
    # Set the aspect of the plot to be equal, so the circle is not skewed
    ax.set_aspect('equal', 'box')
    
    # Set plot limits to show the entire circle
    padding = radius * 0.2
    ax.set_xlim(center[0] - radius - padding, center[0] + radius + padding)
    ax.set_ylim(center[1] - radius - padding, center[1] + radius + padding)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Circle through Three Points')
    plt.grid(True)
    plt.show()


def main():
    # x1, y1 = 3, 3
    # x2, y2 = 6, 6
    # x3, y3 = 10, -4
    x1, y1 = 3, 3
    x2, y2 = 6, 6
    x3, y3 = 5, 6

    radius, center = calculate_circle_radius_and_center(x1, y1, x2, y2, x3, y3)
    print(f"The radius of the circle is: {radius}")
    print(f"The center of the circle is: {center}")

    # Plot the circle
    plot_circle(radius, center, [(x1, y1), (x2, y2), (x3, y3)])

if __name__ == '__main__':
     main()