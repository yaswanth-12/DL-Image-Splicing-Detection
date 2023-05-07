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
```

### Results:

> ##### --- 1 ---
>> - blockiness score:  13.74049886662767
>> - edge score:  0.019286138156882567
>> - combined score:  0.26500115948626957
>> - The image is likely authentic.
> ##### --- 2 ---
>> - blockiness score:  79.17019451935279
>> - edge score:  0.027724460659898476
>> - combined score:  2.1949509433883065
>> - The image may have been spliced.
> ##### --- 3 ---
>> - blockiness score:  96.32394717261904
>> - edge score:  0.0429203869047619
>> - combined score:  4.134261080842656
>> - The image may have been spliced.
> ##### --- 4 ---
>> - blockiness score:  0.06900367647058823
>> - edge score:  196.66945055751174
>> - combined score:  13.570915137918888
>> - The image may have been spliced.
> ##### --- 5 ---
>> - blockiness score:  5.240933333333333
>> - edge score:  0.007404
>> - combined score:  0.0388038704
>> - The image is likely authentic.
> ##### --- 6 ---
>> - blockiness score:  160.56043248935057
>> - edge score:  0.13191574839302111
>> - combined score:  21.180449614139825
>> - The image may have been spliced.
> ##### --- 7 ---
>> - blockiness score:  155.08857913095602
>> - edge score:  0.12834308999081726
>> - combined score:  19.90454746795227
>> - The image may have been spliced.