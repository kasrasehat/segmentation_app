import os
import time
import torch
import argparse
import logging
import uvicorn
import datetime
import asyncio
import gc
import json

import PIL
from PIL import Image, ImageOps
from io import BytesIO
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Response
import torch
import numpy as np

from modules.segmentation import Segmentor


PIL.Image.MAX_IMAGE_PIXELS = 933120000

resolution = {
    'tiny': 256,
    'low': 512,
    'medium': 768,
    'high': 1024
}

def setup_minio(args):
    from minio import Minio

    min_client = Minio(
        args.minioserver,
        access_key=args.miniouser,
        secret_key=args.miniopass,
        secure=args.miniosecure  # Set to False if using an insecure connection (e.g., for local development)
    )

    return min_client


@torch.inference_mode()
def inference(img_org):
    seg_image, sam_dict = segmentor.run_sam(img_org, segmentor, anime_style_chk=False)
    return seg_image, sam_dict


app = FastAPI()  # cal this object by uvicorn api


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

origins = [
    # "http://192.168.1.39:1025"
    # "http://localhost",
    # "http://localhost:1025",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def test_gpu_cuda():
    logging.info('test gpu and cuda:')
    logging.info('\tcuda is available: %s', torch.cuda.is_available())
    logging.info('\tdevice count: %s', torch.cuda.device_count())
    logging.info('\tcurrent device: %s', torch.cuda.current_device())
    logging.info('\tdevice: %s', torch.cuda.device(0))
    logging.info('\tdevice name: %s', torch.cuda.get_device_name())

def get_image(bucket_name, object_name, local_file_path, client):
    client.fget_object(bucket_name, object_name, local_file_path)
    logging.info(f"\timage downloaded: {local_file_path}")

def put_image(bucket_name, object_name, local_file_path, client):
    client.fput_object(bucket_name, object_name, local_file_path)
    logging.info(f"\timage uploaded: {object_name}")

def read_image(object_name, timestamp, beforebucket, client):
    file_name, file_extension = os.path.splitext(object_name)

    temp_save_before_path = f'./{file_name}-{timestamp}'
    local_before_path = temp_save_before_path + file_extension

    get_image(beforebucket, object_name, local_before_path, client)

    img = Image.open(local_before_path).convert("RGB")
    img = ImageOps.exif_transpose(img)

    return img, local_before_path

def runner(object_name: str,
           res_mode: str, debug_mode: bool,
           beforebucket: str, afterbucket: str, 
           client, after_name: str):
    
    timestamp = datetime.datetime.now().strftime('[%Y-%m-%d--%H-%M-%S]')
    logging.info(timestamp)
    
    torch.cuda.empty_cache()
    file_name, file_extension = os.path.splitext(object_name)

    temp_save_before_path = f'./{file_name}-{timestamp}'
    local_before_path = temp_save_before_path + file_extension

    get_image(beforebucket, object_name, local_before_path, client)

    img = Image.open(local_before_path).convert("RGB")
    img = ImageOps.exif_transpose(img)

    logging.info(f'\timage shape: {img.size}', )
    logging.info(f'\tstart inference')
    logging.info(f'\tresolution mode: {res_mode}')
    start = time.time()

    new_img, sam_dict = inference(np.array(img))
    
    end = time.time()

    logging.info(f'\tdone, time: {round(end - start, 4)}')

    local_after_path = after_name
    local_after_path = os.path.basename(local_after_path)
    Image.fromarray(new_img).save(local_after_path)

    # assert img.shape == enhanced_img.shape
    put_image(afterbucket, after_name, local_after_path, client)

    # remove local files
    os.remove(local_before_path)
    os.remove(local_after_path)

    torch.cuda.empty_cache()
    gc.collect()

    if debug_mode:
        # Create an in-memory file-like object
        img_buffer = BytesIO()

        # Save the PIL image to the in-memory buffer
        Image.fromarray(new_img).save(img_buffer, format="JPEG")

        # Set the appropriate media type for the image
        media_type = "image/jpeg"  # Adjust this based on your image format
        img_buffer.seek(0)

        return StreamingResponse(img_buffer, media_type=media_type)
    else:
        output_dict = {}
        for item in sam_dict['sam_masks']:
            true_coords = np.argwhere(item['segmentation'])
            area_number = item['area']
            output_dict[area_number] = true_coords.tolist()

        json_dump = json.dumps(output_dict, cls=NumpyEncoder)
        return json_dump

@app.post("/compute_sam/")
async def compute_sam(object_name: str, 
                             res_mode: str = 'high', 
                             beforebucket: str = '',
                             afterbucket: str = '',
                             debug_mode: bool = False,
                             after_name: str = '',
                             env: str = ''
                             ):
    """
        Perform computation and controlnet processing for an image.

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
        response = await loop.run_in_executor(
                    None,
                    runner,
                    object_name,
                    res_mode,
                    debug_mode, 
                    beforebucket,
                    afterbucket,
                    client,
                    after_name
                    )

        logging.info("\tINFO: POST /compute_sam HTTP/1.1 200 OK")

        return response

    except Exception as e:
        torch.cuda.empty_cache()
        logging.error(f'\tERROR: /compute_sam HTTP:/500, {e}')
        raise HTTPException(status_code=500, detail="Internal Server Error")

def setup_file_logging(file_path, log_level=logging.INFO) -> None:
    """
    setup file logging
    """
    # Set up logging
    logging.basicConfig(level=log_level, format='%(message)s')

    # # Create a file handler and set its level to the desired logging level
    # file_handler = logging.FileHandler(file_path)
    # file_handler.setLevel(log_level)

    # # Create a logging formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # # Set the formatter for the file handler
    # file_handler.setFormatter(formatter)

    # # Add the file handler to the root logger
    # logging.getLogger('').addHandler(file_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup an controlnet API for ID service.")

    parser.add_argument("--server", '-s', type=str, help="Host ip address.")
    parser.add_argument("-port", "-p", type=int, help="Port number.")
    parser.add_argument(
        "--checkpoint_name", "-c", help="checkpoint name.",
        required=False)

    parser.add_argument(
        "--enable_model_cpu_offload", "-e", help="Enable Model CPU Offload",
        action="store_true",
        default=True,
        required=False)

    args = parser.parse_args()

    # args.server = "localhost"
    # args.port = 1025
    
    # args.checkpoint = "/home/mzeinali/projects/models/id/sdxl-v1.0-base-canny/"
    # args.checkpoint_oneformer = "/home/mzeinali/projects/models/oneformer"

    # args.enable_model_cpu_offload = True

    # prepare logging
    timestamp_log = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    setup_file_logging(f'log_id_{timestamp_log}.txt', logging.INFO)

    test_gpu_cuda()

    # global client
    # client = setup_minio(args)

    global segmentor

    sam_checkpoint = '../../models/sam_vit_h_4b8939.pth'
    import pudb; pu.db
    segmentor = Segmentor(sam_checkpoint, 'cuda')
    logging.info('\model segmentation is loaded.')

    uvicorn.run(app, host=args.server, port=args.port)
