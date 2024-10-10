import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon

# Define the parameters for the two normal distributions
mu1, sigma1 = 2.5, 0.4  # Mean and standard deviation for the first distribution
mu2, sigma2 = 4, 0.4  # Mean and standard deviation for the second distribution
scale = 0.5 # Define the parameter for the exponential distribution (rate λ = 1/scale)


# Create a range of x values
x = np.linspace(-5, 7, 1000)

# Compute the probability density functions (PDFs) of both distributions
pdf1 = norm.pdf(x, mu1, sigma1)
pdf2 = norm.pdf(x, mu2, sigma2)
# pdf2 = expon.pdf(x, scale=scale)  # Exponential PDF with specified scale (inverse of rate)
# Create a random distribution and smooth it using a moving average filter
random_pdf = np.random.rand(len(x))  # Random values
window_size = 50  # Window size for smoothing
# random_pdf = np.convolve(random_pdf, np.ones(window_size) / window_size, mode='same')

# Normalize the custom PDF to ensure it integrates to 1
# random_pdf /= np.trapz(random_pdf, x)
# pdf2 = random_pdf


# Compute the product of the two PDFs (element-wise multiplication)
pdf_product = pdf1 * pdf2

# Normalize the resulting product PDF to ensure it forms a valid distribution
pdf_product /= np.trapz(pdf_product, x)

# Plot the two distributions and their product
plt.figure(figsize=(10, 6))

plt.plot(x, pdf1, label=f'N({mu1}, {sigma1}²)', color='blue', linestyle='--')
plt.plot(x, pdf2, label=f'N({mu2}, {sigma2}²)', color='green', linestyle='--')
plt.plot(x, pdf_product, label='Product of N(0, 1²) and N(2, 1²)', color='red')

# Correctly fill the area under the product distribution
# plt.fill_between(x, pdf_product, color='red', alpha=0.2, label='Combined Distribution')

plt.title('Multiplication of Two Normal Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

plt.grid(True)
plt.show()


# Define the parameters for the normal distribution
mu1, sigma1 = 1, 1  # Mean and standard deviation for the normal distribution

# Create a range of x values
x = np.linspace(-5, 7, 1000)

# Compute the PDF of the normal distribution
pdf1 = norm.pdf(x, mu1, sigma1)

# Define parameters for the three peaks in the custom PDF
peak_positions = [0, 2, 5]  # Positions of the three peaks; adjust these as needed
peak_widths = [0.3, 0.5, 0.4]  # Widths (standard deviations) of the peaks; adjust these as needed
peak_heights = [1, 0.8, 1.2]  # Heights of the peaks; adjust these as needed

# Create the multi-peak PDF by summing three Gaussian functions
pdf2 = sum(peak_heights[i] * norm.pdf(x, peak_positions[i], peak_widths[i]) for i in range(3))

# Normalize the custom PDF to ensure it integrates to 1
pdf2 /= np.trapz(pdf2, x)

# Compute the product of the normal PDF and the custom multi-peak PDF
pdf_product = pdf1 * pdf2

# Normalize the resulting product PDF to form a valid distribution
pdf_product /= np.trapz(pdf_product, x)

# Plot the normal distribution, custom three-peak distribution, and their product
plt.figure(figsize=(10, 6))

plt.plot(x, pdf1, label=f'Normal N({mu1}, {sigma1}²)', color='blue', linestyle='--')
plt.plot(x, pdf2, label='Custom Three-Peak Distribution', color='green', linestyle='--')
plt.plot(x, pdf_product, label='Product of Normal and Three-Peak Distribution', color='red')

# Fill the area under the product distribution
# plt.fill_between(x, pdf_product, color='red', alpha=0.2, label='Combined Distribution')

plt.title('Multiplication of Normal and Three-Peak Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

plt.grid(True)
plt.show()


