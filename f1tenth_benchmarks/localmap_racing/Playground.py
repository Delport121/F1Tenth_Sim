import math

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

if __name__ == '__main__':
     main()