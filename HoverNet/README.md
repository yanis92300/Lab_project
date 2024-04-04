All the files have been created from https://github.com/vqdang/hover_net. The environment information has been shared at lenght, the same needs to be followed.
Dataset : 
Pannuke data set has been picked up from the kaggle dataset 
https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification

File "hovernet_training.pynb" - steps followed
1. Set-Up Configuration for Training
2. Copies the data to work folder
3. Set-Up Environment using conda
4. Train the model
5. hovernet_fast_pannuke_type_tf2pytorch.tar is needed for this file which can be found at the github link of hover_net shared above.

File "fasthovenet-checkpoint.pynb" - Runs the pretrained model of hovernet using inference and checkpoints
1. Checkpoints for hovernet can be found the git hub link shared above
2. Input images for the inference needs to be 256x256 and .png files.
3. Segmented files can be found in the pred directory of copied hovernet folder
4. checkpoint01 - needs to be loaded which can be found at github link
5. !python run_infer.py \
--gpu='1' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=64 \
--model_mode=fast \
--model_path=../checkpoint/hovernet_pannuke \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=../hover-net/inference/imgs1/ \
--output_dir=../hover-net/inference/pred/ \
--mem_usage=0.1 \
--draw_dot \
--save_qupath
Parameters needs to be kept the same
6. .mat and .json files are needed to extract the instance details and the boundary box coordinates. These details are needed for postprocessing of hovernet images on SAM

File "postprocess-hovernet-sam.pynb" - Does the postprocessing using the hovernet segmented images on SAM
1. kagglehub.download('segment-anything/pytorch/vit-b/1') - Model for SAM used.
2. Dataset links -
   https://www.kaggle.com/datasets/deeptisammeta/hovernet-overlay - contains the 256x256 images from Pannuke dataset
   https://www.kaggle.com/datasets/deeptisammeta/hovernet-overlay-02 - contains segmented images from hovernet used to postprocess on SAM
   https://www.kaggle.com/datasets/deeptisammeta/images-hover - 256 x 256 images from pannuke but .npy files
4. Segmented images from hoverenet with the instance details and boundary boxes details.
5. masks from the original Panunuke dataset to extract the groundtruth
6. requires 256x256 size images in .png
7. Generating Object Masks with SAM
8. Calculates the IUC and the dice score for images
9. Specified predictions using points and boundary box
10. Calculating the number of contours from the SAM segmented image
11. Calculating the Hausdorff distance for hover net and SAM images
