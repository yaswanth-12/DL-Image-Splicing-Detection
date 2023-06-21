```
import cv2
import numpy as np
import pywt
# Function to calculate the score for Chromatic Aberration


def compute_chromatic_aberration_score(image):
    # Convert the image to the LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into individual channels
    l, a, b = cv2.split(lab)

    # Calculate the average value of the a and b channels
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    # Calculate the chromatic aberration score
    chromatic_aberration_score = abs(a_mean - b_mean)

    return chromatic_aberration_score


# Function to calculate the score for Compression Artifacts
def compute_compression_artifacts_score(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a wavelet transform to the image
    coeffs = pywt.dwt2(gray, 'haar')
    
    # Calculate the standard deviation of the wavelet coefficients
    cA, (cH, cV, cD) = coeffs
    std_dev = np.std(cA)
    
    # Calculate the compression artifacts score
    compression_artifacts_score = 1 - std_dev
    
    return compression_artifacts_score


# Function to calculate the score for Gradient Analysis
def compute_gradient_analysis_score(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient using the Sobel operator
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Calculate the magnitude of the gradient
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Calculate the gradient analysis score
    gradient_analysis_score = np.mean(gradient_mag)

    return gradient_analysis_score


# Function to calculate the score for Quantization Tables
def compute_quantization_tables_score(image):
    # Convert the image to the YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Split the YUV image into individual channels
    y, u, v = cv2.split(yuv)

    # Calculate the average value of the U and V channels
    u_mean = np.mean(u)
    v_mean = np.mean(v)

    # Calculate the quantization tables score
    quantization_tables_score = abs(u_mean - v_mean)

    return quantization_tables_score


# Function to calculate the score for Fourier Analysis
def compute_fourier_analysis_score(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Fourier transform to the image
    fft = np.fft.fft2(gray)

    # Shift the zero-frequency component to the center of the spectrum
    shifted_fft = np.fft.fftshift(fft)

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(shifted_fft)

    # Calculate the Fourier analysis score
    fourier_analysis_score = np.mean(magnitude_spectrum)

    return fourier_analysis_score


def createLBPHistogram(gray_image, num_points, radius):
    # Calculate LBP for each pixel
    lbp = np.zeros_like(gray_image)
    for i in range(radius, gray_image.shape[0] - radius):
        for j in range(radius, gray_image.shape[1] - radius):
            center = gray_image[i, j]
            code = 0
            for k in range(num_points):
                angle = 2 * np.pi * k / num_points
                x = int(round(i + radius * np.cos(angle)))
                y = int(round(j - radius * np.sin(angle)))
                neighbor = gray_image[x, y]
                code |= (neighbor > center) << k
            lbp[i, j] = code

    # Compute histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(
        0, 2 ** num_points + 1), range=(0, 2 ** num_points))
    hist_norm = cv2.normalize(
        hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_norm


def compute_texture_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_hist = createLBPHistogram(gray, num_points=8, radius=1)
    texture_score = lbp_hist.sum()
    return texture_score


def compute_blockiness_score(image, block_size):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the variance of pixel intensities in small blocks
    variances = []
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            block = gray[i:i+block_size, j:j+block_size]
            variances.append(np.var(block))

    # Compute the average variance of the blocks
    blockiness_score = np.mean(variances)
    return blockiness_score


def compute_edge_score(image, threshold1, threshold2):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Compute the ratio of the number of edge pixels to the total number of pixels
    edge_score = np.count_nonzero(
        edges) / float(edges.shape[0] * edges.shape[1])
    return edge_score


def compute_lighting_score(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Calculate the Laplacian of the blurred image
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Calculate the standard deviation of the Laplacian
    std_dev = np.std(laplacian)

    # Calculate the lighting score
    lighting_score = 1 / (1 + np.exp(-std_dev))
    return lighting_score


def compute_color_histogram_score(image):
    # Calculate the color histogram of the image
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalize the histogram
    hist_norm = cv2.normalize(hist, None, alpha=0, beta=1,
                              norm_type=cv2.NORM_MINMAX)

    # Calculate the color histogram score
    color_score = hist_norm.sum()
    return color_score


def compute_noise_score(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the difference between adjacent pixels in each direction
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the standard deviation of the differences
    dx_std = np.std(dx)
    dy_std = np.std(dy)

    # Calculate the noise score
    noise_score = 1 / (1 + np.exp(-(dx_std + dy_std)))
    return noise_score


def detect_splicing(image_path):
    image = cv2.imread(image_path)

    # Compute the blockiness score of the image
    blockiness_score = compute_blockiness_score(image, block_size=2)
    # Compute the edge score of the image
    edge_score = compute_edge_score(image, threshold1=100, threshold2=200)

    # Compute the lighting inconsistency score
    lighting_score = compute_lighting_score(image)

    # Compute the texture analysis score
    texture_score = compute_texture_score(image)

    # Compute the color histogram score
    color_score = compute_color_histogram_score(image)

    # Compute the noise score
    noise_score = compute_noise_score(image)

    # Compute the chromatic aberration score
    chromatic_aberration_score = compute_chromatic_aberration_score(image)

    # Compute the compression artifacts score
    compression_artifacts_score = compute_compression_artifacts_score(image)

    # Compute the gradient analysis score
    gradient_analysis_score = compute_gradient_analysis_score(image)

    # Compute the quantization tables score
    quantization_tables_score = compute_quantization_tables_score(image)

    # Compute the Fourier analysis score
    fourier_analysis_score = compute_fourier_analysis_score(image)

    # Calculate the combined score
    combined_score = (
        blockiness_score *
        edge_score *
        lighting_score *
        texture_score *
        color_score *
        noise_score *
        chromatic_aberration_score *
        compression_artifacts_score *
        gradient_analysis_score *
        quantization_tables_score *
        fourier_analysis_score
    )
    
    combined_score1 = (
        blockiness_score +
        edge_score +
        lighting_score +
        texture_score +
        color_score +
        noise_score +
        chromatic_aberration_score +
        compression_artifacts_score +
        gradient_analysis_score +
        quantization_tables_score +
        fourier_analysis_score
    )

    # Print the scores
    print("blockiness score: ", blockiness_score)
    print("edge score: ", edge_score)
    print("lighting score: ", lighting_score)
    print("texture score: ", texture_score)
    print("color histogram score: ", color_score)
    print("noise score: ", noise_score)
    print("chromatic_aberration score: ", chromatic_aberration_score)
    print("compression artifact score: ", compression_artifacts_score)
    print("gradient analysis score: ", gradient_analysis_score)
    print("quantization table score: ", quantization_tables_score)
    print("fourier analysis score: ", fourier_analysis_score)
    
    # print the final scores
    print("combined score: ", combined_score)
    print("combined score using additon: ", combined_score1)

    # Set the threshold for classification
    threshold = -200000000
    
    # Classify the image as spliced or authentic based on the combined score
    if combined_score < threshold:
        print("The image is classified as spliced.")
    else:
        print("The image is classified as authentic.")
    print("\n")


print("--- 1 ---")
detect_splicing("./dog.jpg")
print("--- 2 ---")
detect_splicing("./14_000000281103.tif")
print("--- 3 ---")
detect_splicing("./108_000000560272.tif")
print("--- 4 ---")
detect_splicing("./0_000000195755.tif")
print("--- 5 ---")
detect_splicing("./dog.jfif")
print("--- 6 ---")
detect_splicing("./Golfer_swing.jpg")
print("--- 7 ---")
detect_splicing("./Golfer_swing (1).jpg")
print("--- 8 ---")
detect_splicing("./car.jpg")
print("--- 9 ---")
detect_splicing("./eagle.jpg")

```
