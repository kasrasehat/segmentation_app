import os
import argparse
import datetime
from io import BytesIO
import asyncio
import logging
import gc
from time import time
import json
from skimage.morphology import erosion, dilation
from fastapi import FastAPI, UploadFile, File, Form
import torch
import uvicorn
import numpy as np
import cv2
import uuid
from PIL import Image
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from typing import List

from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode
from utils import *
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
)

resolution = {
    'tiny': 256,
    'low': 512,
    'medium': 768,
    'high': 1024
}

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# def setup_minio(args):
#     from minio import Minio
#
#     min_client = Minio(
#         args.minioserver,
#         access_key=args.miniouser,
#         secret_key=args.miniopass,
#         secure=args.miniosecure  # Set to False if using an insecure connection (e.g., for local development)
#     )
#     return min_client

def test_gpu_cuda():
    logging.info('test gpu and cuda:')
    logging.info('\tcuda is available: %s', torch.cuda.is_available())
    logging.info('\tdevice count: %s', torch.cuda.device_count())
    logging.info('\tcurrent device: %s', torch.cuda.current_device())
    logging.info('\tdevice: %s', torch.cuda.device(0))
    logging.info('\tdevice name: %s', torch.cuda.get_device_name())


def setup_file_logging(file_path, log_level=logging.INFO):
    # Set up logging
    logging.basicConfig(level=log_level)


def get_bounding_box(mask):
    """
    Get the bounding box of the object in a mask array.

    Parameters:
    mask (np.ndarray): A 2D NumPy array representing the mask,
                       where 'True' values indicate the object.

    Returns:
    tuple: Coordinates of the top-left and bottom-right corners
           of the bounding box as ((min_x, min_y), (max_x, max_y)).
    """
    # Find the indices of the True values
    true_indices = np.argwhere(mask)

    # Determine min and max for x (columns) and y (rows)
    min_y, min_x = true_indices.min(axis=0)
    max_y, max_x = true_indices.max(axis=0)

    # The bounding box is defined by the top-left and bottom-right corners
    bounding_box = [min_x, min_y, max_x, max_y]
    return bounding_box


def resize_image_with_height(pil_image, new_height):
    # Calculate the aspect ratio
    width, height = pil_image.size
    aspect_ratio = width / height

    # Calculate the new width based on the aspect ratio
    new_width = int(new_height * aspect_ratio)

    # Resize the image while preserving the aspect ratio
    resized_image = pil_image.resize((new_width, new_height))

    # Return the resized image
    return resized_image


# def put_image(bucket_name, object_name, local_file_path, client, logging):
#     client.fput_object(bucket_name, object_name, local_file_path)
#     logging.info(f"\timage uploaded: {object_name}")
#
# def get_image(bucket_name, object_name, local_file_path, client, logging):
#     client.fget_object(bucket_name, object_name, local_file_path)
#     logging.info(f"\timage downloaded: {local_file_path}")

def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]

    masks, sinfo, labels = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to('cpu'), segments_info, alpha=0.5
    )
    return masks, sinfo, labels


def setup_cfg(args, dataset, backbone):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_oneformer_config(cfg)
    add_dinat_config(cfg)
    dataset = "ade20k"
    cfg_path = "deploy/" + CFG_DICT[backbone][dataset]
    cfg.merge_from_file(cfg_path)
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
    else:
        cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = args.checkpoint_oneformer
    cfg.freeze()
    return cfg


def get_bounding_box_v2(mask):
    """
    Get the bounding box of the object in a mask array, returned as a tuple of a list.

    Parameters:
    mask (np.ndarray): A 2D NumPy array representing the mask,
                       where 'True' values indicate the object.

    Returns:
    tuple of list: Coordinates of the bounding box as a list in the format [min_x, min_y, max_x, max_y].
    """
    # Find the indices of the True values
    true_indices = np.argwhere(mask)

    # Determine min and max for x (columns) and y (rows)
    min_y, min_x = true_indices.min(axis=0)
    max_y, max_x = true_indices.max(axis=0)

    # The bounding box coordinates are combined into a single list
    bounding_box = [min_x, min_y, max_x, max_y]

    # Return the list as a tuple
    return bounding_box


def calculate_modified_iou(mask1, mask2):
    """
    Calculate a modified IoU metric of two masks, using the area of the smaller mask as the denominator.

    Parameters:
    mask1, mask2 (np.ndarray): Two 2D NumPy arrays representing the masks.

    Returns:
    float: The modified IoU value.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    area_mask1 = mask1.sum()
    area_mask2 = mask2.sum()
    smaller_mask_area = min(area_mask1, area_mask2)

    if smaller_mask_area == 0:
        return 0  # To avoid division by zero if one of the masks is empty
    return intersection / smaller_mask_area


def merge_masks(masks: List = None, labels: List = None,
                bboxes: List = None, sinfo: List = None, threshold: float = 0.9):
    """
    Merge masks based on a modified IoU greater than the given threshold and
    keep the mask with the larger area.

    Parameters:
    masks (list of np.ndarray): A list containing the mask arrays.
    threshold (float): The IoU threshold for merging masks.

    Returns:
    list of np.ndarray: The list of merged masks.
    """
    merged = masks.copy()
    i = 0
    allowed_labels = ['bed', 'sofa', 'chair', 'armchair', 'cushion', 'pillow', 'swivel chair']
    disallowed_labels = ['wall', 'floor', 'vast', 'flower', 'lamp', 'plant', 'ceiling', 'rug']
    while i < len(merged) - 1:
        j = i + 1
        while j < len(merged):
            if labels[i] not in disallowed_labels and labels[j] not in disallowed_labels and \
                    (labels[i] in allowed_labels or labels[j] in allowed_labels):
                iou = calculate_boundingbox_iou(bbox1=bboxes[i], bbox2=bboxes[j])
                if iou >= threshold:
                    # Determine which mask has the smaller area
                    area_i = merged[i].sum()
                    area_j = merged[j].sum()

                    # Merge masks i and j
                    if area_i >= area_j:
                        merged[i] = np.logical_or(merged[i], merged[j])
                        del merged[j]
                        del labels[j]
                        del sinfo[j]
                        del bboxes[j]
                    else:
                        merged[j] = np.logical_or(merged[i], merged[j])
                        del merged[i]
                        del labels[i]
                        del sinfo[i]
                        del bboxes[i]
                        # Since we have removed the i-th mask, we need to adjust the index i
                        if i != 0:
                            i -= 1
                        break  # Exit the inner loop since we have altered the list
                else:
                    j += 1
            else:
                j += 1

        i += 1
    return merged, labels, sinfo


def calculate_boundingbox_iou(bbox1, bbox2):
    """
    Calculate a modified IoU metric of two bounding boxes, using the area of the
    intersection as the numerator and the area of the smaller bounding box as the denominator.

    Parameters:
    bbox1, bbox2 (list of int): Each list contains [min_x, min_y, max_x, max_y] coordinates of a bounding box.

    Returns:
    float: The modified IoU value.
    """
    # Unpack the coordinates
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2

    # Calculate the coordinates of the intersection rectangle
    inter_min_x = max(min_x1, min_x2)
    inter_max_x = min(max_x1, max_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_y = min(max_y1, max_y2)

    # Calculate the intersection area
    intersection_area = max(0, inter_max_x - inter_min_x) * max(0, inter_max_y - inter_min_y)

    # Calculate each bounding box's area
    area_bbox1 = (max_x1 - min_x1) * (max_y1 - min_y1)
    area_bbox2 = (max_x2 - min_x2) * (max_y2 - min_y2)

    # Calculate the area of the smaller bounding box
    smaller_bbox_area = min(area_bbox1, area_bbox2)

    # Calculate modified IoU
    if smaller_bbox_area == 0:
        return 0  # Avoid division by zero
    return intersection_area / smaller_bbox_area


def multi_dilation(image, kernel, iterations):
    for i in range(iterations):
        image = dilation(image, kernel)
    return image


def multi_erosion(image, kernel, iterations):
    for i in range(iterations):
        image = erosion(image, kernel)
    return image


from PIL import Image


def extract_object_with_array(image, mask, background_color='white'):
    """
    Extracts an object from a 3-channel image using a 1-channel mask, leaving the object unaltered,
    and sets the surrounding background to a specified color. The output image will be a 3-channel (RGB) image.

    Parameters:
    - image (numpy.ndarray): The image array of shape (height, width, 3).
    - mask (numpy.ndarray): The mask array of shape (height, width), where non-zero (or white) pixels represent the object.
    - background_color (str): Background color for the output image ('white', 'black', or 'light red').

    Returns:
    - PIL.Image: Image with the object extracted and the specified background color, as a 3-channel RGB image.
    """
    # Ensure the image has 3 channels
    if image.shape[2] != 3:
        raise ValueError("Image must have 3 channels (RGB).")

    # Prepare the background color
    if background_color == 'white':
        bg_color = np.array([255, 255, 255], dtype=np.uint8)
    elif background_color == 'black':
        bg_color = np.array([0, 0, 0], dtype=np.uint8)
    elif background_color == 'light red':
        bg_color = np.array([100, 100, 255], dtype=np.uint8)  # Light red
    else:
        raise ValueError("Unsupported background color. Choose 'white', 'black', or 'light red'.")

    # Initialize the output image as a copy of the original image
    output_image = np.copy(image)

    # Apply the mask to set the background pixels
    # Where mask is 0, set the pixel to the background color
    for c in range(3):  # Iterate over the RGB channels
        output_image[:, :, c] = np.where(mask > 128, bg_color[c], output_image[:, :, c])
    # output_np[mask_expanded > 0] = image_rgba[mask_expanded > 0]
    # Convert the output array back to a PIL Image
    output_image_pil = Image.fromarray(output_image[:, :, ::-1])

    return output_image_pil


def segment_image_runner(
        image_org,
        res_mode,
        base_name,
        im_name
):
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S.%f')
    # logging.info(timestamp)

    # file_name, file_extension = os.path.splitext(object_name)
    #
    # temp_save_before_path = f'./{file_name}-{timestamp}'
    # local_before_path = temp_save_before_path + file_extension

    # get_image(beforebucket, object_name, local_before_path, client, logging)

    # image_org = Image.open(local_before_path).convert("RGB")
    # contents = image_file.read()
    # image_org = Image.open(BytesIO(contents)).convert("RGB")
    w_org, h_org = image_org.size
    image = resize_image_with_height(image_org, resolution[res_mode])
    w_resize, h_resize = image.size
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    masks, sinfo, labels = panoptic_run(image, predictor, metadata)
    masks, sinfo, labels = list(masks), list(sinfo), list(labels)
    bboxes = []
    for mask, label, info in zip(masks, labels, sinfo):
        bboxes.append(get_bounding_box_v2(mask))

    # for bbox, label in zip(bboxes, labels):
    #     if label == 'lamp' or label == 'bed':
    #         cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    #         # Put the label text above the bounding box
    #         label_position = (bbox[0], bbox[1] - 10)  # Position for the label to be above the box
    #         if bbox[1] < 20:  # If the bounding box is too close to the top, put the label below
    #             label_position = (bbox[0], bbox[3] + 20)
    #
    #         cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    # # Show the image with the bounding box
    # cv2.imshow('Bounding Box', image)
    # cv2.waitKey(0)  # Wait for a key press to show the next image
    #
    # cv2.destroyAllWindows()  # Close all OpenCV windows
    # bboxes = tuple(bboxes)
    # os.remove(local_before_path)
    torch.cuda.empty_cache()
    gc.collect()
    # mask_tmp = np.zeros((h_org, w_org), dtype=np.uint8)
    # response = {}
    response = []
    p = 0
    masks, labels, sinfo = merge_masks(masks, labels, bboxes, sinfo)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for mask, label, info in zip(masks, labels, sinfo):
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w_org, h_org))
        image = cv2.resize(image, (w_org, h_org))
        mask = multi_dilation(mask, kernel, 4)
        mask = multi_erosion(mask, kernel, 3)
        # mask = multi_dilation(mask, kernel, 3)
        # mask = cv2.bitwise_not(mask)
        uuid_ = str(uuid.uuid4())
        label = label.split(" ")[0].split(",")[0]
        local_after_path = base_name + '/' + '_' + label + str(p) + '.jpg'
        cv2.imwrite(local_after_path, mask)
        for color in ['white', 'black', 'light red']:
            path = base_name + '/' + '_' + label + str(p) + '_' + color + '.jpg'
            output = extract_object_with_array(image, mask, color)
            output.save(path)
        p += 1

        response.append(
            {
                "tag": label,
                "object_id": uuid_,
            }
        )

    return response
    #
    # else:
    #     point_dict = {}
    #     for label, mask in zip(labels, masks):
    #         service_type = service_type.lower()
    #         if (service_type == "furniture" and label in ['floor', 'wall', 'ceiling']) or \
    #         (service_type == "wall" and label != "wall") or \
    #         (service_type == "ceiling" and label != "ceiling") or \
    #         (service_type == "floor" and label != "floor"):
    #             continue
    #
    #         mask = (mask*255).astype(np.uint8)
    #         mask = cv2.resize(mask, (w_org, h_org))
    #
    #         point_dict[label] = get_white_pixel_coordinates(mask)
    #
    #     point_dict = {key: ','.join(','.join(map(str, sublist)) for sublist in value) for key, value in point_dict.items()}
    #
    #     return json.dumps(point_dict, default=str)


#
# @app.post("/mask_points/")
# async def sagment_image(image_file: UploadFile = File(...),
#                         uuid_: str = '',
#                         env: str = ''
#                         ):
#     try:
#         # args.miniosecure = bool(os.getenv(f'{env}_MINIO_SECURE'))
#         # args.miniouser = os.getenv(f'{env}_MINIO_ACCESS_KEY')
#         # args.miniopass = os.getenv(f'{env}_MINIO_SECRET_KEY')
#         # args.minioserver = os.getenv(f'{env}_MINIO_ADDRESS')
#         #
#         # args.minioserver = "192.168.32.33:9000"
#         # args.miniouser = "test_user_chohfahe7e"
#         # args.miniopass = "ox2ahheevahfaicein5rooyahze4Zeidung3aita6iaNahXu"
#         # args.miniosecure = False
#         #
#         # client = setup_minio(args)
#         #
#         local_before_path = f"{uuid_}.png"
#
#         # get_image(beforebucket, local_before_path, local_before_path, client, logging)
#
#         # mask = Image.open(local_before_path).convert("RGB")
#         contents = await image_file.read()
#         mask = Image.open(BytesIO(contents)).convert("RGB")
#
#         point_dict = ','.join(','.join(map(str, sublist)) for sublist in get_white_pixel_coordinates(np.array(mask)))
#
#         return json.dumps({"points":point_dict}, default=str)
#
#     except Exception as e:
#         torch.cuda.empty_cache()
#         logging.error(f'/mask_points HTTP:/500, {e}')
#         raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/segment_image/")
async def sagment_image(image_file: UploadFile = File(...),
                        res_mode: str = 'medium'):
    """
        Perform computation and instruct pix2pix processing for an image.

    Parameters:
    - **object_name** (str): The name of the image object.
    - **prompt** (str): The prompt to use for the controlnet processing.
    - **res_mode** (str): The resolution mode, tiny: 256, low: 512, medium: 768, high: 1024.
    - **debug_mode** (bool, optional): If True, returns a streaming response of the processed image for debugging purposes.
        Defaults to False.

    Returns:
    - Optional[StreamingResponse]: If debug_mode is True, returns a streaming response containing the processed image.
        Otherwise, returns None.

    Raises:
    - Any exceptions raised during image processing or file operations will be propagated.

    **Note:**
    This function follows the controlnet processing workflow for an image. It retrieves the image file, performs
    inference using the provided prompt, saves the processed image, and uploads it to the specified output bucket.
    If debug_mode is enabled, it returns a streaming response of the processed image for debugging purposes.

    """

    try:
        # Extract filename and separate name and extension
        filename = image_file.filename
        base_name, extension = os.path.splitext(filename)
        im_name = base_name
        base_name = '/home/kasra/PycharmProjects/reimagine-segmentation/data' + "/" + base_name
        # Create a directory with the base name of the image
        if not os.path.exists(base_name):
            os.makedirs(base_name)

            
        contents = await image_file.read()
        image_file = Image.open(BytesIO(contents)).convert("RGB")
        # Save the image in the created directory
        image_save_path = os.path.join(base_name, filename)
        image_file.save(image_save_path)
        loop = asyncio.get_event_loop()
        tic = time()
        response = await loop.run_in_executor(
            None,
            segment_image_runner,
            image_file,
            res_mode,
            base_name,
            im_name,
        )
        logging.info(f"time: {time() - tic}")

        logging.info("POST /compute_segment HTTP/1.1 200 OK")

        # if debug_mode:
        return response

    except Exception as e:
        torch.cuda.empty_cache()
        logging.error(f'/compute_segment HTTP:/500, {e}')
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup a segmentation service.")

    parser.add_argument("--server", '-s', type=str, help="Host ip address.")
    parser.add_argument("-port", "-p", type=int, help="Port number.")
    parser.add_argument(
        "--checkpoint_oneformer", "-co", help="Oneformer segmentation checkpoint.",
        required=False)

    args = parser.parse_args()

    args.server = "0.0.0.0"
    args.port = 4800
    args.checkpoint_oneformer = '/home/kasra/PycharmProjects/reimagine-segmentation/pretrained_models/250_16_swin_l_oneformer_ade20k_160k.pth'

    # prepare logging
    timestamp_log = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    setup_file_logging(f'log_segment_{timestamp_log}.txt', logging.INFO)

    test_gpu_cuda()

    # gl
    #
    # obal client
    # client = setup_minio(args)

    # Loading a single model for all three tasks
    global model, processor, device, predictor, metadata, dataset

    backbone = "Swin-L"
    dataset = "ADE20K (150 classes)"
    cfg = setup_cfg(args, dataset, backbone)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    PREDICTORS[backbone][dataset] = DefaultPredictor(cfg)
    METADATA[backbone][dataset] = metadata
    predictor = PREDICTORS[backbone][dataset]
    print(predictor)
    metadata = METADATA[backbone][dataset]
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uvicorn.run(app, host=args.server, port=args.port)
