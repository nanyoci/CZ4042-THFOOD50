import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import os
import random
import numpy as np
import cv2


# Load models
model_names = ["ViT-Original", "ViT-Food-101", "ViT-No-Pretraining", "VIT-Original-Data-Aug"]
model_dirs = ["./original_model", "./food_model", "./nopretrain_model", "./original_model_data_aug"]
feature_extractors = []
models = []

for model_dir in model_dirs:
    feature_extractors.append(ViTFeatureExtractor.from_pretrained(model_dir))
    models.append(ViTForImageClassification.from_pretrained(model_dir))

assert len(feature_extractors) == len(models)


# Helper functions
def get_attention_map(image, feature_extractor, model):
    inputs = feature_extractor(images=image, return_tensors="pt")
    rescaled_image = (inputs["pixel_values"].reshape(3, 224, 224).transpose(0, 2).transpose(0, 1) + 1) / 2

    model_output = model(output_attentions=True, **inputs)
    logits = model_output.logits
    att_mat = model_output.attentions

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    # Mask & attention maps
    output_mask = cv2.resize((mask - mask.min()) / (mask.max() - mask.min()), (224, 224))
    mask = cv2.resize((mask - mask.min()) / (mask.max() - mask.min()), (224, 224))[..., np.newaxis]
    output_att_map = (mask * np.array(rescaled_image))
    
    # Predicted class
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return rescaled_image, output_mask, output_att_map, predicted_class

def plot_attention_map(model_outputs):
    fig, axarr = plt.subplots(3, 4, figsize=(16, 12))

    for i, model_output in enumerate(model_outputs):
        original_img, mask, att_map, _ = model_output
        axarr[0][i].set_title(f'{model_names[i]}: Original')
        axarr[1][i].set_title(f'{model_names[i]}: Attention')
        axarr[2][i].set_title(f'{model_names[i]}: Image + Attention')

        axarr[0][i].imshow(original_img)
        axarr[1][i].imshow(mask)
        axarr[2][i].imshow(att_map)

        axarr[0][i].axis("off")
        axarr[1][i].axis("off")
        axarr[2][i].axis("off")


# Generate attention maps for 20 random images
test_dir = Path('../THFOOD50-v1/test/')

for i in range(20):
    # Get an image
    class_idx = random.randint(0, models[0].config.num_labels - 1)
    class_name = models[0].config.id2label[class_idx]
    folder = test_dir / class_name
    image_path = random.choice(os.listdir(folder))
    image = Image.open(folder / image_path)
    
    # Get model outputs for each model
    model_outputs = []
    print(f"attention_maps/attention_map_{i + 1}.png")
    for j in range(len(models)):
        model_outputs.append(get_attention_map(image, feature_extractors[j], models[j]))
        print(f"{model_names[j]}: {class_name}({model_outputs[j][3]})")
    print()

    # Plot attention maps
    plot_attention_map(model_outputs)
    plt.savefig(f"attention_maps/attention_map_{i + 1}.png")
