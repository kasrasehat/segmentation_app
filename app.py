import os
import argparse
import datetime
from io import BytesIO
import asyncio
import logging
import gc
from time import time
import json

from fastapi import FastAPI, UploadFile, File
import torch
import uvicorn
import numpy as np
import cv2
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


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_minio(args):
    from minio import Minio

    min_client = Minio(
        args.minioserver,
        access_key=args.miniouser,
        secret_key=args.miniopass,
        secure=args.miniosecure  # Set to False if using an insecure connection (e.g., for local development)
    )
    return min_client


def test_gpu_cuda():
    logging.info('test gpu and cuda:')
    logging.info('\tcuda is available: %s', torch.cuda.is_available())
    logging.info('\tdevice count: %s', torch.cuda.device_count())
    logging.info('\tcurrent device: %s', torch.cuda.current_device())
    logging.info('\tdevice: %s', torch.cuda.device(0))
    logging.info('\tdevice name: %s', torch.cuda.get_device_name())


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


def put_image(bucket_name, object_name, local_file_path, client):
    client.fput_object(bucket_name, object_name, local_file_path)
    logging.info(f"\timage uploaded: {object_name}")

def get_image(bucket_name, object_name, local_file_path, client):
    client.fget_object(bucket_name, object_name, local_file_path)
    logging.info(f"\timage downloaded: {local_file_path}")

def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    
    masks, sinfo, labels = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to('cpu'), segments_info, alpha=0.5
    )
    return masks, sinfo, labels


def setup_cfg(dataset, backbone):
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
    cfg.MODEL.WEIGHTS = MODEL_DICT[backbone][dataset]
    cfg.freeze()
    return cfg

def segment_image_runner(
            image_id,
            res_mode,
            image_file,
            afterbucket,
            client):
    # Load the image
    image_org = Image.open(BytesIO(image_file.file.read())).convert("RGB")
    w_org , h_org = image_org.size
    image = resize_image_with_height(image_org, int(res_mode))
    w_resize, h_resize = image.size
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    masks, sinfo, labels = panoptic_run(image, predictor, metadata)

    response = []
    for i, (mask, info, label) in enumerate(zip(masks, sinfo, labels)):

        mask = np.where(mask,255,0)
        coordinates = np.argwhere(mask == 255)
        center_y, center_x = coordinates[len(coordinates)//2]
        response.append({
            "id":str(i),
            "title": label,
            "point": {
                "x": center_x*100/w_resize,
                "y": center_y*100/h_resize}
            })

        local_after_path = f"{image_id}_{str(i)}_{label}_mask.png"
        local_after_path = os.path.basename(local_after_path)
        cv2.imwrite(local_after_path, mask)
        put_image(afterbucket, local_after_path, local_after_path, client)
        os.remove(local_after_path)
        
    torch.cuda.empty_cache()
    gc.collect()

    return response


def mask_image_runner(
            image_id,
            beforebucket,
            afterbucket,
            selected_objects_id,
            segmented_response,
            client):

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S.%f')
    logging.info(timestamp)

    segmented_response = json.loads(segmented_response)
    mask_images = []
    for selected_object_id in selected_objects_id.split(','):

        selected_object_label = [item["title"] for item in segmented_response if item["id"] == selected_object_id]
        local_before_path = image_id + '_' + selected_object_id + '_' + selected_object_label[0] + '_mask.png'
        get_image(beforebucket, local_before_path, local_before_path, client)
        mask_images.append(cv2.imread(local_before_path))
        os.remove(local_before_path)

    unified_mask = mask_images[0]

    for mask in mask_images[1:]:
        unified_mask = np.logical_or(unified_mask, mask)
    
    print(unified_mask.shape)
    unified_mask = unified_mask.astype(np.uint8) * 255
    local_after_path = f"{image_id}_mask_seletected_{selected_objects_id}.png"
    local_after_path = os.path.basename(local_after_path)
    print(local_after_path)

    cv2.imwrite(local_after_path, unified_mask)
    put_image(afterbucket, local_after_path, local_after_path, client)
    os.remove(local_after_path)
        
    torch.cuda.empty_cache()
    gc.collect()

    return local_after_path



@app.post("/mask_image")
async def mask_image(image_id: str = '', beforebucket: str = '',
                     afterbucket: str = '', debug_mode: bool = False, 
                     selected_objects_id: str = '', segmented_dict: str = '',
                     env: str = ''):
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
        args.miniosecure = bool(os.getenv(f'{env}_MINIO_SECURE'))
        args.miniouser = os.getenv(f'{env}_MINIO_ACCESS_KEY')
        args.miniopass = os.getenv(f'{env}_MINIO_SECRET_KEY')
        args.minioserver = os.getenv(f'{env}_MINIO_ADDRESS')

        args.minioserver = "192.168.32.33:9000"
        args.miniouser = "test_user_chohfahe7e"
        args.miniopass = "ox2ahheevahfaicein5rooyahze4Zeidung3aita6iaNahXu"
        args.miniosecure = False       

        client = setup_minio(args)
        loop = asyncio.get_event_loop()
        tic = time()
        response = await loop.run_in_executor(
                None,
                mask_image_runner,
                image_id,
                beforebucket,
                afterbucket,
                selected_objects_id,
                segmented_dict,
                client
                )
        logging.info(f"time: {time()-tic}")

        logging.info("POST /compute_mask HTTP/1.1 200 OK")

        if debug_mode:
            return response

    except Exception as e:
        torch.cuda.empty_cache()
        logging.error(f'/compute_mask HTTP:/500, {e}')
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/segment_image")
async def sagment_image(image_file: UploadFile = File(...), image_id: str = '',
                        res_mode: str = 'low', afterbucket: str = '',
                        debug_mode: bool = False, env: str = '', after_name: str = ''):
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
        args.miniosecure = bool(os.getenv(f'{env}_MINIO_SECURE'))
        args.miniouser = os.getenv(f'{env}_MINIO_ACCESS_KEY')
        args.miniopass = os.getenv(f'{env}_MINIO_SECRET_KEY')
        args.minioserver = os.getenv(f'{env}_MINIO_ADDRESS')

        args.minioserver = "192.168.32.33:9000"
        args.miniouser = "test_user_chohfahe7e"
        args.miniopass = "ox2ahheevahfaicein5rooyahze4Zeidung3aita6iaNahXu"
        args.miniosecure = False       

        client = setup_minio(args)
        loop = asyncio.get_event_loop()
        tic = time()
        response = await loop.run_in_executor(
                None,
                segment_image_runner,
                image_id,
                res_mode,
                image_file,
                afterbucket,
                client
                )
        logging.info(f"time: {time()-tic}")

        logging.info("POST /compute_segment HTTP/1.1 200 OK")

        if debug_mode:
            return response

    except Exception as e:
        torch.cuda.empty_cache()
        logging.error(f'/compute_segment HTTP:/500, {e}')
        raise HTTPException(status_code=500, detail="Internal Server Error")



def setup_file_logging(file_path, log_level=logging.INFO):
    # Set up logging
    logging.basicConfig(level=log_level)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup a segmentation service.")

    parser.add_argument("--server", '-s', type=str, help="Host ip address.")
    parser.add_argument("-port", "-p", type=int, help="Port number.")

    args = parser.parse_args()

    # args.server = "localhost"
    # args.port = 1024
    
    # args.checkpoint = "/home/mzeinali/projects/instructpix2pix/2023-07-18__00-10-39/diffusers_checkpoint"
    
    args.minioserver = "192.168.32.33:9000"
    args.miniouser = "test_user_chohfahe7e"
    args.miniopass = "ox2ahheevahfaicein5rooyahze4Zeidung3aita6iaNahXu"
    
    # args.beforebucket = "test00before"
    # args.afterbucket = "test00after"

    # prepare logging
    timestamp_log = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    setup_file_logging(f'log_segment_{timestamp_log}.txt', logging.INFO)

    test_gpu_cuda()

    # global client
    # client = setup_minio(args)

    # Loading a single model for all three tasks
    global model, processor, device, predictor, metadata, dataset

    backbone = "Swin-L"
    dataset = "ADE20K (150 classes)"
    cfg = setup_cfg(dataset, backbone)
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

    uvicorn.run(app, host='0.0.0.0', port=1234)
