```
import cv2
import numpy as np

def detect_splicing(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Compute the blockiness score of the image
    blockiness_score = compute_blockiness_score(image, block_size=2)
    # blockiness_score = compute_blockiness_score(image, block_size=16)

    # Compute the edge score of the image
    edge_score = compute_edge_score(image, threshold1=100, threshold2=200)

    # Combine the blockiness and edge scores
    splicing_score = blockiness_score * edge_score
    print(splicing_score)
    # Check if the splicing score is above a threshold
    if splicing_score > 0.5:
        print("The image may have been spliced.")
    else:
        print("The image is likely authentic.")

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
    print(blockiness_score)
    
    return blockiness_score

def compute_edge_score(image, threshold1, threshold2):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Compute the ratio of the number of edge pixels to the total number of pixels
    edge_score = np.count_nonzero(edges) / float(edges.shape[0] * edges.shape[1])
    print(edge_score)
    return edge_score

detect_splicing("../images/dog.jpg")
detect_splicing("../images/dog.jfif")
detect_splicing("../images/1.jpg")
detect_splicing("../images/0_000000195755.tif")
```