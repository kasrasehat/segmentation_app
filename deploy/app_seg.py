import os
import argparse
import datetime
from io import BytesIO
import asyncio
import logging
import gc
from time import time
import json

from fastapi import FastAPI,UploadFile, File, Form
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
    w_org , h_org = image_org.size
    image = resize_image_with_height(image_org, resolution[res_mode])
    w_resize, h_resize = image.size
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    masks, sinfo, labels = panoptic_run(image, predictor, metadata)

    # os.remove(local_before_path)
    torch.cuda.empty_cache()
    gc.collect()


    # mask_tmp = np.zeros((h_org, w_org), dtype=np.uint8)
    # response = {}
    response = []
    p = 0
    for mask, label, info in zip(masks, labels, sinfo):
        # service_type = service_type.lower()
        # if service_type == "furniturewall" and label in ['wall', 'ceiling'] or \
        # (service_type == "furniture" and label in ['floor', 'wall', 'ceiling']) or \
        # (service_type == "wall" and label != "wall") or \
        # (service_type == "ceiling" and label != "ceiling") or \
        # (service_type == "floor" and label != "floor"):
        #     continue


        mask = (mask*255).astype(np.uint8)
        mask = cv2.resize(mask, (w_org, h_org))
        mask = cv2.bitwise_not(mask)

        # if service_type == "furniturewall":
        #     mask_tmp += mask

        # else:
        #     ar_area = info['area'] * 100 / mask.size
        #     if ar_area < 0.5:
        #         continue
        uuid_ = str(uuid.uuid4())
        label = label.split(" ")[0].split(",")[0]
        local_after_path = base_name + '/' + '_' + label + str(p) + '.jpg'
        cv2.imwrite(local_after_path, mask)
        p += 1

            # put_image(afterbucket, local_after_path, local_after_path, client, logging)

            # os.remove(local_after_path)

        response.append(
            {
                "tag":label,
                "object_id":uuid_,
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
        logging.info(f"time: {time()-tic}")

        logging.info("POST /compute_segment HTTP/1.1 200 OK")

        # if debug_mode:
        return response

    except Exception as e:
        torch.cuda.empty_cache()
        logging.error(f'/compute_segment HTTP:/500, {e}')
        raise HTTPException(status_code=500, detail="Internal Server Error")

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
    args.port = 4500
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
