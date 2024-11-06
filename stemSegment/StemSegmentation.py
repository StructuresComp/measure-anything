# import torch
import numpy as np
import cv2
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import SAM

# class SegmentAnything:
#     def __init__(self, sam_model_path):
#         self.sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#         self.sam_model_path = sam_model_path
#         self.predictor = SAM2ImagePredictor(build_sam2(self.sam_model_cfg, self.sam_model_path))  # SAM predictor object for handling prompts
#
#     def _inference(self, image, positive_prompts, negative_prompts=None, visualize=False):
#         # Convert image to the format expected by SAM2
#         self.predictor.set_image(image)
#
#         # Swap x and y for OpenCV coordinates if necessary
#         # if positive_prompts.shape[0] > 1:
#         #     positive_prompts[:, [0, 1]] = positive_prompts[:, [1, 0]]
#         # if negative_prompts is not None:
#         #     if negative_prompts.shape[0] > 1:
#         #         negative_prompts[:, [0, 1]] = negative_prompts[:, [1, 0]]
#         #     prompts = np.vstack([positive_prompts, negative_prompts])
#         # else:
#         #     prompts = positive_prompts
#
#         # Stack prompts
#         prompts = np.vstack([positive_prompts, negative_prompts]) if negative_prompts is not None else positive_prompts\
#
#         # Create labels for positive (1) and negative (0) prompts
#         labels = np.zeros(prompts.shape[0], dtype=np.int8)
#         labels[:positive_prompts.shape[0]] = 1
#
#         # Run SAM2 prediction with prompts and labels
#         with torch.no_grad():
#             masks, scores, logits = self.predictor.predict(
#                 point_coords=prompts,
#                 point_labels=labels,
#                 multimask_output=False  # Set to False to return only one mask
#             )
#
#         # Extract the final mask
#         # final_mask = masks[0].cpu().numpy()
#         final_mask = masks[0]
#
#         # Visualize overlay with points
#         if visualize:
#             overlay_image = image.copy()
#             overlay_image[final_mask == 1] = (0, 255, 0)  # Color mask in green
#             for prompt in positive_prompts:
#                 cv2.circle(overlay_image, (prompt[0], prompt[1]), 5, (0, 0, 255), -1)
#             if negative_prompts is not None:
#                 for prompt in negative_prompts:
#                     cv2.circle(overlay_image, (prompt[0], prompt[1]), 5, (255, 0, 0), -1)
#             cv2.imwrite("mask_overlay_pos_neg_points.png", overlay_image)
#
#         return final_mask


class StemSegmentation:
    def __init__(self):
        self.model = SAM("sam2.1_l.pt")

    def inference(self, image, positive_prompts, negative_prompts=None):
        # Stack prompts
        prompts = np.vstack([positive_prompts, negative_prompts]) if negative_prompts is not None else positive_prompts\

        # Create labels for positive (1) and negative (0) prompts
        labels = np.zeros(prompts.shape[0], dtype=np.int8)
        labels[:positive_prompts.shape[0]] = 1

        # Run SAM2 prediction with prompts and labels
        masks = self.model(image, points=[prompts], labels=[labels])
        final_mask = masks[0].masks.data[0].cpu().numpy()

        return final_mask

