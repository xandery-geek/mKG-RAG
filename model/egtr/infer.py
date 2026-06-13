import os
import json
import itertools
import torch
import argparse
from functools import partial
from tqdm import tqdm
from glob import glob
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from .model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from .model.egtr import DetrForSceneGraphGeneration

from tools.initialization import load_vqa_dataset
from dataset.utils import image_path_collate_fn, vqa_collate_fn
from dataset.image import ImagePathDataset


def color_generator():
    """
    This is a generator that yields colors in BGR format.
    It loops through a set of predefined colors and also
    yields randomly generated colors when the predefined ones are exhausted.
    """
    # Predefined colors in BGR format
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (0, 255, 255),    # Cyan
        (255, 0, 255),    # Magenta
        (255, 192, 203),  # Pink
        (165, 42, 42),    # Brown
        (255, 165, 0),    # Orange
        (128, 0, 128),     # Purple
        (0, 0, 128),       # Navy
        (128, 0, 0),      # Maroon
        (128, 128, 0),    # Olive
        (70, 130, 180),   # Steel Blue
        (173, 216, 230),  # Light Blue
        (255, 192, 0),    # Gold
        (255, 165, 165),  # Light Salmon
        (255, 20, 147),   # Deep Pink
    ]
    for color in itertools.cycle(colors):
        yield color


def load_model(args):
    # feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=args.min_size, max_size=args.max_size
    )

    # model
    config = DeformableDetrConfig.from_pretrained(args.artifact_path)
    model = DetrForSceneGraphGeneration.from_pretrained(
        args.architecture, config=config, ignore_mismatched_sizes=True
    )

    ckpt_path = sorted(
        glob(f"{args.artifact_path}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k[6:]] = state_dict.pop(k)  # "model."

    model.load_state_dict(state_dict)
    model = model.eval()
    model = model.to(args.device)

    return feature_extractor, model


def load_label_info(label_path):
    with open(label_path, 'r') as f:
        label_info = json.load(f)
    return label_info


def infer(feature_extractor, model, batch_images, args):
    """
    Inference for Scene Graph Generation
    Output:
    - valid_obj_classes: object classes, list of class ids
    - valid_obj_boxes: object bounding boxes, list of [center_x, center_y, width, height]
    - valid_triplets: triplets, list of [subject index in valid_obj_classes, object index in valid_obj_classes, relation id]
    """
    # inference image
    encodings = feature_extractor(batch_images, return_tensors="pt")

    # output
    with torch.no_grad():
        batch_outputs = model(
            pixel_values=encodings['pixel_values'].to(model.device), 
            pixel_mask=encodings['pixel_mask'].to(model.device), 
            output_attention_states=True
        )

    # postprocess
    batch_obj_classes, batch_obj_boxes, batch_triplets = [], [], []

    for idx in range(len(batch_images)):
        obj_classes, obj_boxes, triplets = postprocess(batch_outputs, idx, args)
        batch_obj_classes.append(obj_classes)
        batch_obj_boxes.append(obj_boxes)
        batch_triplets.append(triplets)
    
    return batch_obj_classes, batch_obj_boxes, batch_triplets


def postprocess(outputs, idx, args):

    def get_inclusion_matrix(obj_bbox, self_loop=True):
        N = obj_bbox.size(0)
        
        # Extract coordinates
        x1 = obj_bbox[:, 0].view(N, 1)
        y1 = obj_bbox[:, 1].view(N, 1)
        x2 = obj_bbox[:, 2].view(N, 1)
        y2 = obj_bbox[:, 3].view(N, 1)
        
        # Check if j's bbox is included in i's bbox
        # For j's bbox to be included in i's bbox:
        # x1[i] <= x1[j] and y1[i] <= y1[j] and x2[i] >= x2[j] and y2[i] >= y2[j]
        inclusion_matrix = (
            (x1 <= x1.t()) &  # x1[i] <= x1[j]
            (y1 <= y1.t()) &  # y1[i] <= y1[j]
            (x2 >= x2.t()) &  # x2[i] >= x2[j]
            (y2 >= y2.t())   # y2[i] >= y2[j]
        )
        
        if not self_loop:
            inclusion_matrix.fill_diagonal_(False)
        
        return inclusion_matrix

    def filter_out_overlap_classes(obj_scores, obj_classes, obj_boxes, box_mode='xywh', iou_threshold=0.5, 
                                   filter_by_area=False, obj_areas=None):
        """
        Filters out overlapping objects.
        1. For objects that share the same class and high IoU, keep the one with the highest score.
        2. For a pair of objects (i, j) that share the same class, if i's bbox includes j's bbox, filter out j.
        Args:
            - obj_scores: object scores, tensor of shape (N,)
            - obj_classes: object classes, tensor of shape (N,)
            - obj_boxes: object bounding boxes, tensor of shape (N, 4)
            - iou_threshold: IoU threshold for filtering overlapping objects
            - filter_by_area: whether to filter by area
            - obj_areas: object areas, tensor of shape (N,)
        Returns:
            - obj_scores: filtered object scores
        """
        if box_mode == 'xywh':
            # convert xywh to xyxy
            obj_boxes = torch.cat([
                obj_boxes[:, :2] - obj_boxes[:, 2:] / 2,
                obj_boxes[:, :2] + obj_boxes[:, 2:] / 2
            ], dim=1)

        if filter_by_area and obj_areas is None:
            obj_areas = (obj_boxes[:, 2] - obj_boxes[:, 0]) * (obj_boxes[:, 3] - obj_boxes[:, 1])

        # Calculate IoU matrix
        device = obj_scores.device
        iou_matrix = box_iou(obj_boxes, obj_boxes)
        inclusion_matrix = get_inclusion_matrix(obj_boxes)

        # Iterate over each object
        for i in range(len(obj_scores)):
            # Skip if the score is already 0
            if obj_scores[i] == 0:
                continue

            # Find indices of objects with the same class
            same_class = obj_classes == obj_classes[i]

            # Checking overlapping by IoU
            high_iou = iou_matrix[i] > iou_threshold
            overlapping = same_class & high_iou
            overlapping[i] = False # Exclude itself

            # Set scores of overlapping objects to 0, except the one with the highest score
            if overlapping.any():
                overlapping_indices = torch.where(overlapping)[0]
                all_indices = torch.cat([torch.tensor([i], device=device), overlapping_indices])

                filter_metrics = obj_areas[all_indices] if filter_by_area else obj_scores[all_indices]
                max_idx = all_indices[torch.argmax(filter_metrics)]
                overlapping_indices = overlapping_indices[overlapping_indices != max_idx]
                obj_scores[overlapping_indices] = 0
            
            # Checking overlapping by inclusion
            overlapping = same_class & inclusion_matrix[i]
            overlapping[i] = False
            if overlapping.any():
                overlapping_indices = torch.where(overlapping)[0]
                obj_scores[overlapping_indices] = 0

        return obj_scores
    
    ## Get scores
    # get object scores, classes, boxes
    pred_logits = outputs['logits'][idx]
    obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), -1)
    pred_boxes = outputs['pred_boxes'][idx]

    # get relation scores
    pred_connectivity = outputs['pred_connectivity'][idx]
    pred_rel = outputs['pred_rel'][idx]
    pred_rel = torch.mul(pred_rel, pred_connectivity)
    
    ## Filter out objects and triplets
    # calculate a dynamic threshold based on the mean and std of object scores
    obj_threshold = obj_scores.mean() + args.obj_coeff * obj_scores.std()

    # filter out small boxes
    bbox_areas = (pred_boxes[:, 2] * pred_boxes[:, 3])
    small_bbox_indices = (bbox_areas < args.bbox_threshold).nonzero()[:, 0]
    obj_scores[small_bbox_indices] = 0

    # filter out overlapping objects
    obj_scores = filter_out_overlap_classes(
        obj_scores, pred_classes, pred_boxes, box_mode='xywh', iou_threshold=0.4,
    )

    # get top k objects
    top_k_obj_scores, top_k_obj_indices = obj_scores.topk(args.top_k)

    # get valid objects based on object threshold
    valid_obj_indices = (top_k_obj_scores >= obj_threshold).nonzero()[:, 0]

    # map to the original indices
    valid_obj_indices = top_k_obj_indices[valid_obj_indices]
    valid_obj_classes = pred_classes[valid_obj_indices]
    valid_obj_boxes = pred_boxes[valid_obj_indices]

    # Get the maximum value and index for each relation
    max_values, max_indices = torch.max(pred_rel, dim=2)

    # Calculate a dynamic threshold based on the mean and std of relation scores
    rel_threshold = max_values.mean() + args.rel_coeff * max_values.std()

    # Get relations between valid objects
    max_values = max_values[valid_obj_indices][:, valid_obj_indices]
    max_indices = max_indices[valid_obj_indices][:, valid_obj_indices]

    # For relationships A -> B and B -> A, keep the one with the higher score
    mask = max_values > max_values.t()
    max_values[~mask] = 0

    # Get indices of valid triplets
    valid_tri_indices = (max_values >= rel_threshold).nonzero(as_tuple=True)

    # Extract valid triplets: [subject id, object id, relation id]
    valid_triplets = torch.stack([
        valid_tri_indices[0],
        valid_tri_indices[1],
        max_indices[valid_tri_indices]
    ], dim=1)

    return valid_obj_classes.detach().cpu().numpy(), \
        valid_obj_boxes.detach().cpu().numpy(), \
        valid_triplets.detach().cpu().numpy()


def id2label(label_info, ids, id_type='object'):
    """
    Convert object or relation ids to label names
    """
    assert id_type in ['object', 'relation']
    # check if ids is iterable
    if not hasattr(ids, '__iter__'):
        ids = [ids]
    
    label_map = label_info[id_type]
    return [label_map[str(i)] for i in ids]


def xywh2xyxy(bbox, width_scale=1, height_scale=1):
    x_center, y_center, width, height = bbox
    x_min = min(max((x_center - width/2), 0), 1) * width_scale
    y_min = min(max((y_center - height/2), 0), 1) * height_scale
    x_max = min(max((x_center + width/2), 0), 1) * width_scale
    y_max = min(max((y_center + height/2), 0), 1) * height_scale
    return x_min, y_min, x_max, y_max


def visualize(image, obj_classes, obj_boxes, text_width=50, text_height=15):
    img_width, img_height = image.size
    
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for obj, bbox in zip(obj_classes, obj_boxes):
        # convert ccwh to xyxy
        x_min, y_min, x_max, y_max = xywh2xyxy(bbox, img_width, img_height)
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # draw bounding box and label
        color = next(color_gen)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color) # bbox
        draw.rectangle(((x_min, y_min), (x_min + text_width, y_min + text_height)), fill=color) # label
        draw.text((x_min, y_min), str(obj)) # label text
    
    return image


def output_to_txt(image, obj_classes, obj_boxes, triplets, save_path, label_info):

    label_triples = []
    for triplet in triplets:
        sub, obj, rel = triplet
        # convert object index to class id
        sub, obj = obj_classes[sub], obj_classes[obj]
        label_triples.append([id2label(label_info, sub, id_type='object')[0], 
                            id2label(label_info, obj, id_type='object')[0],
                            id2label(label_info, rel, id_type='relation')[0]])

    label_classes = id2label(label_info, obj_classes, id_type='object')
    with open(save_path, 'w') as f:
        f.write("Total objects: {}\n".format(len(label_classes)))
        for i, obj in enumerate(label_classes):
            f.write(f"{i+1}: {obj}\n")
        f.write("Total triples: {}\n".format(len(label_triples)))
        for i, triple in enumerate(label_triples):
            f.write(f"{i+1}: {triple[0]} -> {triple[2]} -> {triple[1]}\n")
    
    # save image with bounding boxes
    vis_image = visualize(image, label_classes, obj_boxes)
    vis_image.save(save_path.replace('.txt', '.jpg'),)


def output_to_json(image, obj_classes, obj_boxes, triples, save_path, label_info):
    objects, relations = [], []

    for i, (obj, bbox) in enumerate(zip(obj_classes, obj_boxes)):
        x_min, y_min, x_max, y_max = xywh2xyxy(bbox)
        objects.append({
            "id": i,
            "category": id2label(label_info, obj, id_type='object')[0],
            "bbox": [x_min, y_min, x_max, y_max]
        })

    for i, triplet in enumerate(triples):
        sub, obj, rel = triplet
        relations.append({
            "id": i,
            "relation": id2label(label_info, rel, id_type='relation')[0],
            "objects": [int(sub), int(obj)],
        })

    with open(save_path, 'w') as f:
        json.dump({
            "objects": objects,
            "relations": relations
        }, f)


def get_output_path(image_path, output_dir, output_format='json'):
    image_filename = os.path.basename(image_path).split('.')[0]
    if output_format == 'txt':
        output_path = os.path.join(output_dir, f"{image_filename}.txt")
    else:
        output_path = os.path.join(output_dir, f"{image_filename}.json")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Scene Graph Generation')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--dataset', type=str, default='vg', choices=['vg', 'oi'], help='dataset')
    parser.add_argument('--architecture', type=str, default='SenseTime/deformable-detr', help='architecture')
    parser.add_argument('--artifact_path', type=str, required=True, help='model path')
    parser.add_argument('--image_dataset', type=str, default='envqa', help='image dataset')
    parser.add_argument('--image_dataset_split', type=str, default='val', help='image dataset split')
    parser.add_argument('--image_dir', type=str, default="", help='image path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='max workers')
    parser.add_argument('--output_dir', type=str, default='output', help='output path')
    parser.add_argument('--min_size', type=int, default=800, help='min size')
    parser.add_argument('--max_size', type=int, default=1333, help='max size')
    parser.add_argument('--top_k', type=int, default=20, help='top k objects')
    parser.add_argument('--obj_coeff', type=float, default=0.5, help='coefficent for object threshold')
    parser.add_argument('--rel_coeff', type=float, default=0, help='coefficent for relation threshold')
    parser.add_argument('--bbox_threshold', type=float, default=0.1, help='bbox threshold')
    parser.add_argument('--output_format', type=str, default='txt', choices=['txt', 'json'], help='output format')

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Current visible GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")

    feature_extractor, model = load_model(args)
    label_info = load_label_info(f'model/egtr/data/{args.dataset}/label_info.json')

    os.makedirs(args.output_dir, exist_ok=True)
    output_path_func = partial(get_output_path, output_dir=args.output_dir, output_format=args.output_format)
    output_func = output_to_txt if args.output_format == 'txt' else output_to_json

    if args.image_dataset != "" and args.image_dataset_split != "":
        print("Generating scene graphs for images in the dataset")
        dataset = load_vqa_dataset(args.image_dataset, args.image_dataset_split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=vqa_collate_fn)
    
    elif args.image_dir != "":
        print("Generating scene graphs for images in the directory")
        image_files = glob(f"{args.image_dir}/*.jpg")

        unprocessed_files = [f for f in image_files if not os.path.exists(output_path_func(f))]
        dataset = ImagePathDataset(unprocessed_files, transform=None)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=image_path_collate_fn)
    else:
        print("No image directory or dataset specified")
        return
    
    for batch in tqdm(dataloader):
        batch_images = batch[0]
        batch_image_paths = batch[-1]
        batch_size = len(batch_images)

        batch_obj_classes, batch_obj_boxes, batch_triplets = infer(feature_extractor, model, batch_images, args)

        for i in range(batch_size):
            image_path = batch_image_paths[i]

            save_path = get_output_path(image_path, args.output_dir, args.output_format)
            output_func(
                batch_images[i],
                batch_obj_classes[i],
                batch_obj_boxes[i],
                batch_triplets[i],
                save_path,
                label_info
            )


if __name__ == '__main__':
    color_gen = color_generator()
    main()
