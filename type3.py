import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def apply_kmeans(img, k=8):
    # Reshape image to a 2D array of pixels
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(Z)
    centers = np.uint8(kmeans.cluster_centers_)
    clustered_img = centers[labels.flatten()]
    clustered_img = clustered_img.reshape(img.shape)

    # Pseudo-accuracy metric
    inertia = kmeans.inertia_
    max_inertia = np.linalg.norm(Z - np.mean(Z, axis=0))**2
    pseudo_accuracy = (1 - (inertia / max_inertia)) * 100

    return clustered_img, round(pseudo_accuracy, 2)

def cartoonify_image(image_path, k=8):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return None, 0

    # Resize for faster processing
    img = cv2.resize(img, (512, 512))

    # Applying KMeans for color simplification
    simplified, accuracy = apply_kmeans(img, k=k)

    # Apply bilateral filter
    smoothed = cv2.bilateralFilter(simplified, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to grayscale and blur
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    # Detect edges
    edges = cv2.adaptiveThreshold(gray_blurred, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)

    # Combine colors with edges
    cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)

    return cartoon, accuracy

def main():
    image_path = input("Enter the full path to your image: ")
    k = int(input("Enter the number of colors (k) for K-means: "))

    cartoon_image, accuracy = cartoonify_image(image_path, k=k)

    if cartoon_image is not None:
        print(f"K-Means Clustering Accuracy (Color Simplification): {accuracy}%")

        # Display original and cartoonified images side by side
        original_image = cv2.imread(image_path)
        original_resized = cv2.resize(original_image, (512, 512))
        combined = np.hstack((original_resized, cartoon_image))

        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title("Original (left) vs Cartoonified (right)")
        plt.axis('off')
        plt.show()

        # Save output
        cv2.imwrite("cartoonified_image_kmeans.jpg", cartoon_image)
        print("Cartoonified image saved as cartoonified_image_kmeans.jpg")

if __name__ == "__main__":
    main()
