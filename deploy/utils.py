import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# Ignore Future and User Warnings
import warnings
from huggingface_hub import hf_hub_download


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


KEY_DICT = {
            "ADE20K (150 classes)": "ade20k",}

SWIN_CFG_DICT = {
            "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}


DINAT_CFG_DICT = {
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}


CFG_DICT = {"DiNAT-L": DINAT_CFG_DICT,
        "Swin-L": SWIN_CFG_DICT }


PREDICTORS = {
    "DiNAT-L": {
        "ADE20K (150 classes)": None
    },
    "Swin-L": {
        "ADE20K (150 classes)": None
    }
}

METADATA = {
    "DiNAT-L": {
        "ADE20K (150 classes)": None
    },
    "Swin-L": {
        "ADE20K (150 classes)": None
    }
}


def get_white_pixel_coordinates(binary_image):

    white_coordinates = np.where(binary_image==255)
    coordinates = np.transpose((white_coordinates[1],white_coordinates[0]))
    coordinates = coordinates.tolist()
    # coordinates = [[x,y] for x, y in zip(white_coordinates[1], white_coordinates[0])] 
    return coordinates

def show_segmentation_results(predicted_semantic_map):
    # Define the color map for masks
    predicted_semantic_map = np.array(predicted_semantic_map.to('cpu'))
    num_classes = len(np.unique(predicted_semantic_map))
    colormap = plt.cm.get_cmap('hsv', num_classes)

    # Create a random color map for each class
    color_map = [colormap(i) for i in range(num_classes)]

    # Map the predicted semantic map to the corresponding colors
    colored_map = np.zeros((predicted_semantic_map.shape[0], predicted_semantic_map.shape[1], 3))
    for i, class_id in enumerate(np.unique(predicted_semantic_map)):
        mask = predicted_semantic_map == class_id
        colored_map[mask] = color_map[i][:3]  # Assign RGB color
    plt.imshow(colored_map)
    plt.axis('off')
    plt.show()
    return colored_map, np.unique(predicted_semantic_map)
    # Display the segmented image
    # plt.imshow(colored_map)
    # plt.axis('off')
    # plt.show()
