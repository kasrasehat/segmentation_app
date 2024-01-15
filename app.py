import os
import argparse
import datetime
from io import BytesIO
import asyncio
import logging
import gc
from time import time
import json

from fastapi import FastAPI
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
    # logging.info(f"\timage uploaded: {object_name}")

def get_image(bucket_name, object_name, local_file_path, client):
    client.fget_object(bucket_name, object_name, local_file_path)
    # logging.info(f"\timage downloaded: {local_file_path}")

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
    dataset = "coco"
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
            object_name,
            res_mode,
            beforebucket,
            client):

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S.%f')
    logging.info(timestamp)

    torch.cuda.empty_cache()
    file_name, file_extension = os.path.splitext(object_name)

    temp_save_before_path = f'./{file_name}-{timestamp}'
    local_before_path = temp_save_before_path + file_extension

    get_image(beforebucket, object_name, local_before_path, client)

    image_org = Image.open(local_before_path).convert("RGB")
    w_org , h_org = image_org.size
    image = resize_image_with_height(image_org, int(res_mode))
    w_resize, h_resize = image.size
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    masks, sinfo, labels = panoptic_run(image, predictor, metadata)

    point_dict = {}
    for label, mask in zip(labels, masks):
        point_dict[label] = get_white_pixel_coordinates(mask)

    os.remove(local_before_path)    
    torch.cuda.empty_cache()
    gc.collect()

    return json.dumps(point_dict, default=str)


@app.post("/segment_image")
async def sagment_image(object_name: str = '',
                        res_mode: str = '1024', before_bucket: str = '',
                        debug_mode: bool = False, env: str = ''):
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
                object_name,
                res_mode,
                before_bucket,
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
    dataset = "COCO (133 classes)"
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
