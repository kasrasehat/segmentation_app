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

SWIN_MODEL_DICT = {
              "ade20k": hf_hub_download(repo_id="shi-labs/oneformer_ade20k_swin_large", 
                                            filename="250_16_swin_l_oneformer_ade20k_160k.pth")
            }

DINAT_CFG_DICT = {
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

DINAT_MODEL_DICT = {
              "ade20k": hf_hub_download(repo_id="shi-labs/oneformer_ade20k_dinat_large", 
                                            filename="250_16_dinat_l_oneformer_ade20k_160k.pth")
            }

MODEL_DICT = {"DiNAT-L": DINAT_MODEL_DICT,
        "Swin-L": SWIN_MODEL_DICT }

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
    coordinates = []
    height, width = binary_image.shape[:2]

    for y in range(height):
        for x in range(width):
            if binary_image[y][x] == 1:  # Assuming white pixels are represented by 255
                coordinates.append([x, y])

    return coordinates


def point_provider(mask_dict):
    keys = list(mask_dict.keys())

    point_dict = {}
    for i in keys:
        point_dict[i] = get_white_pixel_coordinates(mask_dict[i])
    return point_dict

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


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def tagger(predicted_semantic_map):
    x = np.unique(np.array(predicted_semantic_map.to('cpu')))
    # x = np.array(predicted_semantic_map.to('cpu'))
    # print(x)
    tags = []
    for i in x:
        tags.append(id2label[str(i)])
    return tags


def mask_provider(predicted_semantic_map, tag_list):
    predicted_semantic_map = np.array(predicted_semantic_map.to('cpu'))
    result = {}
    for i in tag_list:
        mask = np.where(predicted_semantic_map == label2id[i], 1, 0)
        result[i] = mask
    return result


def whole_mask_attacher(mask_dict):
    key_list = list(mask_dict.keys())
    image = Image.fromarray(np.uint8(mask_dict[key_list[0]] * 255))

    for i in range(1, len(key_list)):
        pil_image = Image.fromarray(np.uint8(mask_dict[key_list[i]] * 255))
        image = get_concat_v(image, pil_image)

    return image


def specific_mask(mask_dict, tag):
    image = Image.fromarray(np.uint8(mask_dict[str(tag)] * 255))
    return image
