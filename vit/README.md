# Vision Transformer (ViT) Experiments
- This folder contains the code for running the ViT model experiments
- Some sample outputs can be found in the folders:
	- `out/`
	- `attention_maps/`
	- `original_model/`
	- `food_model/`
	- `nopretrain_model/`

## How to Run
1. Install packages: `pip install -r requirements.txt`
2. Run model experiments
	1. ViT-Original: `python vit_model.py --pretrain original --epochs 10`
	2. ViT-Food-101: `python vit_model.py --pretrain food101 --epochs 10`
	3. ViT-No-Pretraining: `python vit_model.py  --epochs 10`
3. Visualise attention maps: `python visualise_attention.py`

## Code Inspiration
- `vit_model.py`: https://github.com/nateraw/huggingpics/blob/main/HuggingPics.ipynb
- `visualise_attention.py`: https://www.kaggle.com/piantic/vision-transformer-vit-visualize-attention-map/notebook
