import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the parameters for the two normal distributions
mu1, sigma1 = 0, 1  # Mean and standard deviation for the first distribution
mu2, sigma2 = 2, 1  # Mean and standard deviation for the second distribution

# Create a range of x values
x = np.linspace(-5, 7, 1000)

# Compute the probability density functions (PDFs) of both distributions
pdf1 = norm.pdf(x, mu1, sigma1)
pdf2 = norm.pdf(x, mu2, sigma2)

# Compute the product of the two PDFs (element-wise multiplication)
pdf_product = pdf1 * pdf2

# Normalize the resulting product PDF to ensure it forms a valid distribution
pdf_product /= np.trapz(pdf_product, x)

# Plot the two distributions and their product
plt.figure(figsize=(10, 6))

plt.plot(x, pdf1, label=f'N({mu1}, {sigma1}²)', color='blue', linestyle='--')
plt.plot(x, pdf2, label=f'N({mu2}, {sigma2}²)', color='green', linestyle='--')
plt.plot(x, pdf_product, label='Product of N(0, 1²) and N(2, 1²)', color='red')

# Ensure the fill_between is correctly implemented
plt.fill_between(x, pdf_product, color='red', alpha=0.2, label='Combined Distribution')

plt.title('Multiplication of Two Normal Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

plt.grid(True)
plt.show()

