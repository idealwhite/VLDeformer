# Pytorch implement of the paper "VLDeformer: Vision Language Decomposed Transformer for Fast Cross-modal Retrieval"
This is a pytorch implementation of the [VLDeformer](https://www.sciencedirect.com/science/article/pii/S0950705122006608) paper. Please remember to give a citation if this paper and codes benefits your research!
```
@article{zhang2022vldeformer,
  title={VLDeformer: Vision--Language Decomposed Transformer for fast cross-modal retrieval},
  author={Zhang, Lisai and Wu, Hongfa and Chen, Qingcai and Deng, Yimeng and Siebert, Joanna and Li, Zhonghua and Han, Yunpeng and Kong, Dejiang and Cao, Zhao},
  journal={Knowledge-Based Systems},
  volume={252},
  pages={109316},
  year={2022},
  publisher={Elsevier}
}
```
# Environment Setup
Install the python packages following [VinVL](https://github.com/microsoft/Oscar/blob/master/INSTALL.md). To achieve the performance in the paper, at least 6 V100 GPU is suggested.

# Dataset and Pre-processed Files
You need to download the dataset COCO and Flickr30k to reproduce the exprements, and also SBU for the large version.

Besides, we use the features extracted by VinVL, which are given in their [download page](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md). You can directly download these features from [coco](https://biglmdiag.blob.core.windows.net/vinvl/image_features/coco_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/) and [flikckr30k](https://biglmdiag.blob.core.windows.net/vinvl/image_features/flickr30k_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/).


If you want to run the model on your customed data, please refer to [Scene Graph](https://github.com/microsoft/scene_graph_benchmark) to extract the features, which is specified by the VinVL repo.
 

# Pre-trained Model Checkpoints
The decomposition is applied on the pre-trained one stream VinVL [model](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md#pre-trained-models), so you need to download it first.
```
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/TASK_NAME' coco_ir --recursive
```

Afterwards, you can run our code to perform decomposition.

You can also directly use our pre-trained checkpoints for [Flickr30k](https://drive.google.com/file/d/1nL1GUj62TssgRO34SoHwKKXVFrXRktrw/view?usp=sharing) and [COCO](https://drive.google.com/file/d/1nL1GUj62TssgRO34SoHwKKXVFrXRktrw/view?usp=sharing).

# Running
Run contrastive_learn.py using following args:
```
"program": "${workspaceFolder}/contrastive_learn.py",
"args": [
    "--model_name_or_path",
    "vinvl/coco_ir/base/checkpoint-1340000",    // Your path to Vinvl checkpoint
    "--data_dir",
    "/raid/data_modal/coco_vinvL/coco_ir/",     // Your path to Vinvl data
    "--img_feat_file",
    "/raid/data_modal/coco_vinvL/model_0060000/features.tsv", // Your path to Vinvl image feature tsv
    "--eval_img_keys_file",         
    "test_img_keys_1k.tsv",         // select the test file in ${data_dir}  
    "--do_train",
    "--do_lower_case",
    "--evaluate_during_training",
    "--num_captions_per_img_val", "5",
    "--per_gpu_train_batch_size", "300",
    "--per_gpu_eval_batch_size", "300",
    "--learning_rate", "7.5e-06",
    "--warmup_steps", "200",
    "--scheduler", "cos",
    "--num_train_epochs", "200",
    "--save_steps", "400",
    "--add_od_labels",
    "--od_label_type",
    "vg",
    "--max_seq_length",
    "35",
    "--max_img_seq_length",
    "70",
    "--output_dir",
    "contrastive_checkpoint/output_infoNCE_coswarmup",
    "--logit_gpu", "2,3,4,5,6,7",      // Contrative learning requires large batch size
    "--contrastive_gpu", "0",
    "--temperature_t", "0.005",
    "--temperature_i", "0.005",
]
```

# Acknowledge
This repo is modified based on the [VinVL](https://github.com/microsoft/Oscar), we thank the authors for sharing their project.