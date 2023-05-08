```
import cv2
import numpy as np


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


def detect_splicing(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Compute the blockiness score of the image
    blockiness_score = compute_blockiness_score(image, block_size=2)

    # Compute the edge score of the image
    edge_score = compute_edge_score(image, threshold1=100, threshold2=200)

    # Compute the lighting inconsistency score
    lighting_score = compute_lighting_score(image)

    # Compute the texture analysis score
    texture_score = compute_texture_score(image)

    print("blockiness score: ", blockiness_score)
    print("edge score: ", edge_score)
    print("lighting score: ", lighting_score)
    print("texture score: ", texture_score)
    # Combine the scores
    combined_score = blockiness_score * edge_score * lighting_score * texture_score
    print("combined score: ", combined_score)

    # Check if the combined score is above a threshold
    if combined_score > 0.5:
        print("The image may have been spliced.")
    else:
        print("The image is likely authentic.")
    print("\n")


def compute_lighting_score(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Calculate the Laplacian of the blurred image
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Calculate the standard deviation of the Laplacian
    std_dev = np.std(laplacian)
    # print(std_dev)

    # Calculate the lighting score
    lighting_score = 1 / (1 + np.exp(-std_dev))
    return lighting_score


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
detect_splicing("./landscape.jpg")
```

### Results:

> ##### --- 1 ---
>> - blockiness score:  13.74049886662767
>> - edge score:  0.019286138156882567
>> - lighting score:  0.817976144490086
>> - texture score:  1
>> - combined score:  0.21676462672198116
>> - The image is likely authentic.

> ##### --- 2 ---
>> - blockiness score:  79.17019451935279
>> - edge score:  0.027724460659898476
>> - lighting score:  0.8525746555325443
>> - texture score:  1
>> - combined score:  1.8713595444701185
>> - The image may have been spliced.

> ##### --- 3 ---
>> - blockiness score:  96.32394717261904
>> - edge score:  0.0429203869047619
>> - lighting score:  0.8512804617122647
>> - texture score:  1
>> - combined score:  3.5194156817387823
>> - The image may have been spliced.

> ##### --- 4 ---
>> - blockiness score:  196.66945055751174
>> - edge score:  0.06900367647058823
>> - lighting score:  0.9190316096120675
>> - texture score:  1
>> - combined score:  12.47209998311037
>> - The image may have been spliced.

> ##### --- 5 ---
>> - blockiness score:  5.240933333333333
>> - edge score:  0.007404
>> - lighting score:  0.7810021782250718
>> - texture score:  1
>> - combined score:  0.03030590730596339
>> - The image is likely authentic.

> ##### --- 6 ---
>> - blockiness score:  160.56043248935057
>> - edge score:  0.13191574839302111
>> - lighting score:  0.8407248781121844
>> - texture score:  1
>> - combined score:  17.806930920208966
>> - The image may have been spliced.

> ##### --- 7 ---
>> - blockiness score:  155.08857913095602
>> - edge score:  0.12834308999081726
>> - lighting score:  0.8380923983197337
>> - texture score:  1
>> - combined score:  16.6818499248851
>> - The image may have been spliced.

> ##### --- 8 ---
>> - blockiness score:  99.16888076241135
>> - edge score:  0.0642930728241563
>> - lighting score:  0.8408356132928988
>> - texture score:  1
>> - combined score:  5.36106030456595
>> - The image may have been spliced.
