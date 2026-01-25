import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.ops import roi_pool
from streamlit_cropper import st_cropper
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set page configuration
st.set_page_config(layout="wide")

st.title("Interactive ROI Pooling Illustration for Fast R-CNN")

st.write("""
This application demonstrates the concept of ROI (Region of Interest) Pooling, a crucial component of object detection models like Fast R-CNN.

**Steps:**
1.  **Upload an Image:** Choose an image you want to analyze.
2.  **Configure Parameters:** Use the sidebar to select the model layer, ROI pool size, and number of channels to display.
3.  **Select a Region of Interest (ROI):** Use the cropping tool to select a part of the image.
4.  **See the Results:** Click the button to see the feature extraction and the fixed-size output from the ROI pooling operation.
""")

# --- Sidebar for User Configuration ---
st.sidebar.title("Configuration")

# Load a pre-trained model (VGG16)
@st.cache_resource
def load_model():
    return models.vgg16(weights=models.VGG16_Weights.DEFAULT)

model = load_model()
features = model.features

# Create a dictionary to map layer names to their index
layer_names = {f"Layer {i} - {layer.__class__.__name__}": i for i, layer in enumerate(features) if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.MaxPool2d)}

# User selects the layer
selected_layer_name = st.sidebar.selectbox(
    "Choose a layer to extract features from:",
    list(layer_names.keys())
)
selected_layer_index = layer_names[selected_layer_name]

# User selects the ROI pooling output size
pool_size_str = st.sidebar.text_input("Enter ROI Pooling Output Size (e.g., 7,7):", "7,7")
try:
    output_size = tuple(map(int, pool_size_str.split(',')))
    if len(output_size) != 2:
        st.sidebar.error("Please enter two comma-separated integers.")
        st.stop()
except ValueError:
    st.sidebar.error("Invalid format. Please use two comma-separated integers.")
    st.stop()


# User selects the number of channels to display
num_channels_to_display = st.sidebar.slider(
    "Number of pooled feature channels to display:",
    min_value=1,
    max_value=512, # VGG16 has up to 512 channels
    value=5
)
# --- End of Sidebar ---


# 1. User uploads an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Select a Region of Interest (ROI)")
        # 2. User selects a region
        cropped_box = st_cropper(image, realtime_update=True, box_color='blue', aspect_ratio=None, return_type='box')
        st.write("Selected Box Coordinates:", cropped_box)

    with col2:
        if cropped_box:
            st.subheader("Selected ROI")
            cropped_img = image.crop((cropped_box['left'], cropped_box['top'], cropped_box['left'] + cropped_box['width'], cropped_box['top'] + cropped_box['height']))
            st.image(cropped_img)


    if st.button("Run Analysis", key="run_analysis"):
        if not cropped_box:
            st.warning("Please select a region of interest on the image first.")
        else:
            with st.spinner("Processing..."):
                # Preprocess the image
                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0)

                # 3. Get the feature map from the selected layer
                feature_extractor = torch.nn.Sequential(*list(features.children())[:selected_layer_index + 1])
                with torch.no_grad():
                    feature_map = feature_extractor(input_batch)

                st.subheader("2. Feature Extraction")
                st.info(f"Feature map extracted from **{selected_layer_name}**.")
                st.write(f"The shape of the feature map is: `{feature_map.shape}`")
                st.write(f"This shape represents `(batch_size, channels, height, width)`.")

                if image.width > 0:
                    spatial_scale = feature_map.shape[3] / image.width
                else:
                    spatial_scale = 1

                # --- ENHANCEMENT 2: Spatial Scale Info ---
                st.info(f"""
                **Spatial Scale Calculation:**
                - Original Image Width: `{image.width}px`
                - Feature Map Width: `{feature_map.shape[3]}px`
                - Scale Factor: `{spatial_scale:.4f}` (approx 1/{1/spatial_scale:.1f})
                
                Your selected ROI `{cropped_box['width']}x{cropped_box['height']}` (in original pixels) becomes approximately `{int(cropped_box['width'] * spatial_scale)}x{int(cropped_box['height'] * spatial_scale)}` in feature map pixels.
                """)

                roi = torch.tensor([[
                    0,
                    cropped_box['left'] * spatial_scale,
                    cropped_box['top'] * spatial_scale,
                    (cropped_box['left'] + cropped_box['width']) * spatial_scale,
                    (cropped_box['top'] + cropped_box['height']) * spatial_scale
                ]], dtype=torch.float32)

                # --- ENHANCEMENT 1: Projected ROI Visualization ---
                st.subheader("2.1 Projection on Feature Map")
                st.write("This shows where your selected region lands on the downsampled feature map.")

                # Visualize the mean activation map
                mean_activation_map = feature_map[0].mean(0).detach().numpy()

                fig_proj, ax_proj = plt.subplots(figsize=(6, 6))
                ax_proj.imshow(mean_activation_map, cmap='jet')

                # Add the bounding box
                # ROI format: [batch_idx, x1, y1, x2, y2]
                roi_x = roi[0, 1].item()
                roi_y = roi[0, 2].item()
                roi_w = roi[0, 3].item() - roi_x
                roi_h = roi[0, 4].item() - roi_y

                rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h, linewidth=2, edgecolor='white', facecolor='none')
                ax_proj.add_patch(rect)
                ax_proj.set_title("Projected ROI on Average Feature Map")
                ax_proj.axis('off')

                st.pyplot(fig_proj)

                # 4. Apply ROI pooling
                pooled_features = roi_pool(feature_map, roi, output_size)

                # --- ENHANCEMENT 3: Grid Visualization on Original ROI ---
                st.subheader("3. Pooling Grid Visualization")
                st.write(f"This shows how the {output_size[0]}x{output_size[1]} grid divides your original/cropped image area.")

                if cropped_box:
                    cropped_img_np = np.array(cropped_img)
                    fig_grid, ax_grid = plt.subplots(figsize=(4, 4))
                    ax_grid.imshow(cropped_img_np)

                    # Draw grid lines
                    img_h, img_w, _ = cropped_img_np.shape
                    step_h = img_h / output_size[0]
                    step_w = img_w / output_size[1]

                    for i in range(1, output_size[0]):
                        ax_grid.axhline(i * step_h, color='white', linestyle='--', linewidth=1)
                    for i in range(1, output_size[1]):
                        ax_grid.axvline(i * step_w, color='white', linestyle='--', linewidth=1)

                    ax_grid.set_title(f"Original ROI with {output_size} Pooling Grid")
                    ax_grid.axis('off')
                    st.pyplot(fig_grid)

                st.subheader("4. ROI Pooling Result")
                st.write(f"The shape of the pooled features is: `{pooled_features.shape}`.")
                st.success(f"Notice that the output spatial dimension is now fixed to **{output_size}**, as you specified, regardless of the original ROI size.")

                # --- ENHANCEMENT 5: Matrix View for Small Sizes ---
                if output_size[0] * output_size[1] <= 100:
                    st.write("#### 'Under the Hood' - Values of the first pooled channel")
                    st.write("These are the actual values resulting from the max/average pooling operation in each bin.")
                    # Use the first channel (index 0)
                    df_values = pooled_features[0, 0].detach().numpy()
                    st.dataframe(df_values)

                st.write(f"### Visualizing the first {num_channels_to_display} channels of the Pooled Features")

                # Visualize the selected number of channels
                cols = st.columns(num_channels_to_display)
                for i in range(min(num_channels_to_display, pooled_features.shape[1])):
                    with cols[i]:
                        # --- NORMALIZATION FIX STARTS HERE ---
                        pooled_map_tensor = pooled_features[0, i].detach()
                        
                        # Perform min-max normalization to scale values to the 0-1 range
                        min_val = torch.min(pooled_map_tensor)
                        max_val = torch.max(pooled_map_tensor)
                        
                        # Avoid division by zero if the map is all the same value
                        if max_val > min_val:
                            normalized_map = (pooled_map_tensor - min_val) / (max_val - min_val)
                        else:
                            normalized_map = torch.zeros_like(pooled_map_tensor)
                        
                        # Convert to numpy for display
                        pooled_map_numpy = normalized_map.numpy()
                        
                        st.image(pooled_map_numpy, caption=f'Channel {i+1}', use_container_width=True)
                        # --- NORMALIZATION FIX ENDS HERE ---
