# Interactive ROI Pooling Demo

This application is an interactive educational tool designed to demonstrate the concept of **Region of Interest (ROI) Pooling**, a critical operation in object detection architectures like Fast R-CNN and Faster R-CNN.

It allows students and users to visualize how variable-sized regions in an image are mapped to fixed-size feature maps, demystifying the "projection" and "pooling" steps.

## Features

-   **Interactive ROI Selection**: Upload any image and draw a bounding box to select a Region of Interest.
-   **Configurable Parameters**:
    -   Select different layers from a pre-trained VGG16 model.
    -   Define custom output grid sizes (e.g., $7 \times 7$, $3 \times 3$).
-   **Educational Visualizations**:
    -   **Feature Map Projection**: See exactly where your original image ROI lands on the downsampled feature map.
    -   **Grid Overlay**: Visualize how the pooling grid divides the original image region.
    -   **Spatial Scale Math**: Explicit calculation showing the scaling factor between image and feature space.
    -   **Matrix View**: For small pooling sizes, inspect the actual numerical values of the pooled features.

## Prerequisites

-   Python 3.8+
-   `pip` package manager

## Installation

1.  Clone the repository or download the source code.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default web browser (usually at `http://localhost:8501`).

## Docker Support

A `Dockerfile.txt` is included for containerized deployment.

1.  **Build the Docker image** (you may need to rename `Dockerfile.txt` to `Dockerfile` first):
    ```bash
    docker build -t roi-pooling-demo -f Dockerfile.txt .
    ```

2.  **Run the container**:
    ```bash
    docker run -p 8080:8080 roi-pooling-demo
    ```
    Access the app at `http://localhost:8080`.

## Acknowledgments

Created for **AAI3001** (Advanced AI) lectures to illustrate Computer Vision concepts.
