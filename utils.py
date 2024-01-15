import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# Ignore Future and User Warnings
import warnings
from huggingface_hub import hf_hub_download


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


KEY_DICT = {"COCO (133 classes)": "coco",}

SWIN_CFG_DICT = {
            "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml"}

SWIN_MODEL_DICT = {
              "coco": hf_hub_download(repo_id="shi-labs/oneformer_coco_swin_large", 
                                            filename="150_16_swin_l_oneformer_coco_100ep.pth"),
            }

DINAT_CFG_DICT = {
             "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",}

DINAT_MODEL_DICT = {
              "coco": hf_hub_download(repo_id="shi-labs/oneformer_coco_dinat_large", 
                                            filename="150_16_dinat_l_oneformer_coco_100ep.pth")
            }

MODEL_DICT = {"DiNAT-L": DINAT_MODEL_DICT,
        "Swin-L": SWIN_MODEL_DICT }

CFG_DICT = {"DiNAT-L": DINAT_CFG_DICT,
        "Swin-L": SWIN_CFG_DICT }


PREDICTORS = {
    "DiNAT-L": {
        "COCO (133 classes)": None
    },
    "Swin-L": {
        "COCO (133 classes)": None
    }
}

METADATA = {
    "DiNAT-L": {
        "COCO (133 classes)": None
    },
    "Swin-L": {
        "COCO (133 classes)": None
}}


def get_white_pixel_coordinates(binary_image):

    white_coordinates = np.where(binary_image==1)
    coordinates = [[x,y] for x, y in zip(white_coordinates[1], white_coordinates[0])] 

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
