device=${1:-"0"}

WORKING_DIR="./working_dir"
DATA_DIR="./data"

EGTR_DATASET="vg"
EGTR_DATA_DIR="./weights/egtr"

if [ $EGTR_DATASET == "vg" ]; then
    ARTIFACT_PATH="$EGTR_DATA_DIR/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0"
elif [ $EGTR_DATASET == "oi" ]; then
    ARTIFACT_PATH="$EGTR_DATA_DIR/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__open_image__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__open_image__finetune/version_0"
else
    echo "Dataset not supported"
    exit
fi

export CUDA_VISIBLE_DEVICES=$device

# script for testing
# python -m model.egtr.infer \
#     --device "cuda" \
#     --dataset $EGTR_DATASET \
#     --artifact_path $ARTIFACT_PATH \
#     --output_dir "$WORKING_DIR/sgg/$EGTR_DATASET" \
#     --image_dataset "" \
#     --image_dataset_split "" \
#     --image_dir "./examples/img" \
#     --output_format "json"

# script for inference on image dataset (images in vqa samples)
python -m model.egtr.infer \
    --device "cuda" \
    --dataset $EGTR_DATASET \
    --artifact_path $ARTIFACT_PATH \
    --output_dir "$DATA_DIR/encyclopedic-vqa/scene_graph/envqa" \
    --image_dataset "envqa" \
    --image_dataset_split "test" \
    --output_format "json"

## script for inference on image directory (images in knowledge base)
python -m model.egtr.infer \
    --device "cuda" \
    --dataset $EGTR_DATASET \
    --artifact_path $ARTIFACT_PATH \
    --output_dir "$DATA_DIR/encyclopedic-vqa/envqa_test/scene_graph" \
    --image_dataset "" \
    --image_dataset_split "" \
    --image_dir "$DATA_DIR/encyclopedic-vqa/kb_images_640/all" \
    --output_format "json"
