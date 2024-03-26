import os
import argparse
import datetime
from io import BytesIO
import asyncio
import logging
import gc
from time import time
import json

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
    cfg_path = CFG_DICT[backbone][dataset]
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


def merge_masks(masks, labels, sinfo, threshold=0.5):
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
    while i < len(merged) - 1:
        j = i + 1
        while j < len(merged):
            iou = calculate_modified_iou(merged[i], merged[j])
            if iou > threshold:
                # Determine which mask has the smaller area
                area_i = merged[i].sum()
                area_j = merged[j].sum()

                # Merge masks i and j
                if area_i >= area_j:
                    merged[i] = np.logical_or(merged[i], merged[j])
                    del merged[j]
                    del labels[j]
                    del sinfo[j]
                else:
                    merged[j] = np.logical_or(merged[i], merged[j])
                    del merged[i]
                    del labels[i]
                    del sinfo[i]
                    # Since we have removed the i-th mask, we need to adjust the index i
                    if i != 0:
                        i -= 1
                    break  # Exit the inner loop since we have altered the list
            else:
                j += 1
        i += 1
    return merged, labels, sinfo


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
    # bboxes = []
    # for mask, label, info in zip(masks, labels, sinfo):
    #     bboxes.append(get_bounding_box_v2(mask))
    #
    # bboxes = tuple(bboxes)
    # os.remove(local_before_path)
    torch.cuda.empty_cache()
    gc.collect()

    # mask_tmp = np.zeros((h_org, w_org), dtype=np.uint8)
    # response = {}
    response = []
    p = 0
    masks, labels, sinfo = merge_masks(masks, labels, sinfo)
    for mask, label, info in zip(masks, labels, sinfo):


        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w_org, h_org))
        mask = cv2.bitwise_not(mask)
        uuid_ = str(uuid.uuid4())
        label = label.split(" ")[0].split(",")[0]
        local_after_path = base_name + '/' + '_' + label + str(p) + '.jpg'
        cv2.imwrite(local_after_path, mask)
        p += 1

        response.append(
            {
                "tag": label,
                "object_id": uuid_,
            }
        )

        # if service_type == "furniturewall":
        #     uuid_ = str(uuid.uuid4())
        #     local_after_path = f"{uuid_}.png"
        #     cv2.imwrite(local_after_path, mask_tmp)

        # put_image(afterbucket, local_after_path, local_after_path, client, logging)

        # response.append(
        #         {
        #             "tag":"furniturewall",
        #             "object_id":uuid_,
        #         }
        #     )
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
