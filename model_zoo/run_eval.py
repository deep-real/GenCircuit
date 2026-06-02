import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.PACS_dataset import PACSDataset

model_dict = {
    'ViT-B_16-in21k':'vit_base_patch16_224_in21k',
    'ViT-S_16':'vit_small_patch16_224_in21k',
    'ViT-Ti_16':'vit_tiny_patch16_224_in21k',
    'ViT-B_16-clip-openai': 'vit_base_patch16_clip_224.openai',
    'ViT-B_16-clip-laion2b': 'vit_base_patch16_clip_224.laion2b',
    'ViT-B_16-mae': 'vit_base_patch16_224.mae',
    'ViT-B_16-dinov2': 'vit_base_patch14_dinov2.lvd142m',
    'ViT-B_16-in1k': 'vit_base_patch16_224.orig_in21k_ft_in1k',
    'ViT-B_16-in21k-hook':'vit_base_patch16_224_in21k',
    'ViT-B_16-scratch': 'vit_base_patch16_224_in21k'
}

def evaluate_model_on_domain(model, dataloader, device=None):
    """
    Evaluate classification model accuracy on the given domain dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the domain.
        device (torch.device or str): Device to run evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Classification accuracy (0.0 - 1.0).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # handle models that return (logits, features) etc.

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# Paths
csv_path = "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/PACS_sweep_results.csv"
output_csv_path = "/output/PACS_sweep_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load previous results
df = pd.read_csv(csv_path)

# Add new columns for cartoon and art
new_columns = ['test_acc_cartoon', 'test_acc_art_paint']
for col in new_columns:
    if col not in df.columns:
        df[col] = None

for idx, row in tqdm(df.iterrows(), total=len(df)):
    checkpoint_path = row['checkpoint']
    model_type = row['model_type']
    model_name = model_dict[model_type]
    model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=7,
            drop_rate = 0.1,
            img_size = 224
        )
    checkpoint = torch.load(row["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    config = timm.data.resolve_model_data_config(model)
    transform_test = timm.data.create_transform(**config, is_training=False)

    # Evaluate on cartoon
    cartoon_dataset = PACSDataset('/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/PACS/PACS/sketch', '/home/yxpengcs/PycharmProjects/assaying-ood/webdata/cartoon/split.json', 'val', transform=transform_test)
    cartoon_loader = DataLoader(cartoon_dataset,
                            # sampler=val_sampler,
                            batch_size=64,
                            num_workers=4,
                            pin_memory=True)
    acc_cartoon = evaluate_model_on_domain(model, cartoon_loader, device)

    # Evaluate on art_paint
    art_dataset = PACSDataset('/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/PACS/PACS/sketch', '/home/yxpengcs/PycharmProjects/assaying-ood/webdata/art_painting/split.json', 'val', transform=transform_test)
    art_loader = DataLoader(art_dataset,
                                # sampler=val_sampler,
                                batch_size=64,
                                num_workers=4,
                                pin_memory=True)
    acc_art = evaluate_model_on_domain(model, art_loader, device)

    # Append new results
    df.at[idx, 'test_cartoon_acc'] = acc_cartoon
    df.at[idx, 'test_art_paint_acc'] = acc_art

# Save updated CSV
df.to_csv(output_csv_path, index=False)
print(f"Updated CSV with cartoon and art-paint OOD accuracy saved to {output_csv_path}")
