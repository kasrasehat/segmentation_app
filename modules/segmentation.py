import gc
import platform
import os
import torch
import numpy as np

from modules.fast_sam import FastSamAutomaticMaskGenerator, fast_sam_model_registry
from modules.mobile_sam import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorMobile
from modules.mobile_sam import sam_model_registry as sam_model_registry_mobile
from modules.segment_anything_fb import SamAutomaticMaskGenerator, sam_model_registry
from modules.segment_anything_hq import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorHQ
from modules.segment_anything_hq import sam_model_registry as sam_model_registry_hq
import modules.inpalib as inpalib



class Segmentor():
    def __init__(self, sam_checkpoint, device):
        self.sam_mask_generator = self.get_sam_mask_generator(sam_checkpoint, device)
        self.sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)

    def run_sam(self, input_image, sam_mask_generator_, anime_style_chk=False):

        if self.sam_dict["sam_masks"] is not None:
            self.sam_dict["sam_masks"] = None
            gc.collect()

        sam_masks = inpalib.generate_sam_masks(input_image, self.sam_mask_generator, anime_style_chk)
        sam_masks = inpalib.sort_masks_by_area(sam_masks)
        sam_masks = inpalib.insert_mask_to_sam_masks(sam_masks, self.sam_dict["pad_mask"])
        seg_image = inpalib.create_seg_color_image(input_image, sam_masks)

        self.sam_dict["sam_masks"] = sam_masks

        return seg_image, self.sam_dict
        
    def get_sam_mask_generator(self, sam_checkpoint, device, anime_style_chk=False):
        """Get SAM mask generator.

        Args:
            sam_checkpoint (str): SAM checkpoint path

        Returns:
            SamAutomaticMaskGenerator or None: SAM mask generator
        """
        if "_hq_" in os.path.basename(sam_checkpoint):
            model_type = os.path.basename(sam_checkpoint)[7:12]
            sam_model_registry_local = sam_model_registry_hq
            SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGeneratorHQ
            points_per_batch = 32
        elif "FastSAM" in os.path.basename(sam_checkpoint):
            model_type = os.path.splitext(os.path.basename(sam_checkpoint))[0]
            sam_model_registry_local = fast_sam_model_registry
            SamAutomaticMaskGeneratorLocal = FastSamAutomaticMaskGenerator
            points_per_batch = None
        elif "mobile_sam" in os.path.basename(sam_checkpoint):
            model_type = "vit_t"
            sam_model_registry_local = sam_model_registry_mobile
            SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGeneratorMobile
            points_per_batch = 64
        else:
            model_type = os.path.basename(sam_checkpoint)[4:9]
            sam_model_registry_local = sam_model_registry
            SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGenerator
            points_per_batch = 64

        pred_iou_thresh = 0.88 if not anime_style_chk else 0.83
        stability_score_thresh = 0.95 if not anime_style_chk else 0.9
        import pudb; pu.db
        if os.path.isfile(sam_checkpoint):
            sam = sam_model_registry_local[model_type](checkpoint=sam_checkpoint)
            if platform.system() == "Darwin":
                if "FastSAM" in os.path.basename(sam_checkpoint):
                    sam.to(device=torch.device("cpu"))
                else:
                    sam.to(device=torch.device("mps"))
            else:
                sam.to(device=device)
            
            sam_mask_generator = SamAutomaticMaskGeneratorLocal(
                model=sam, points_per_batch=points_per_batch, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)
        else:
            sam_mask_generator = None

        return sam_mask_generator
