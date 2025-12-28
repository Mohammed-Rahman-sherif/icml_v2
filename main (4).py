import os
import random
import argparse # <<< CHANGED: Added argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb
from datasets import build_dataset
# Import both data loader builders from our unified utils file
from datasets.utils import build_data_loader, build_graph_data_loader
import clip
from utils import *
# Import the correct HGT model
from hgtmodel import HGTImageFeatureExtractor

wandb_log = True
# <<< CHANGED: Removed global BATCH_SIZE, will get from wandb.config

# <<< CHANGED: get_arguments is no longer needed, using argparse in main
# def get_arguments():
# ...

def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    # Ensure cache_values is in the correct dtype from the start
    cache_values = cache_values.to(val_features.dtype)

    print("\n-------- Searching hyperparameters on the val set. --------")
    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    # <<< CHANGED: Using wandb.config for hyperparameters
    beta, alpha = wandb.config.init_beta, wandb.config.init_alpha
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    # cache_values is now passed with the correct dtype
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, image_model, train_loader_F):
    """
    An enhanced version that combines Zero-Shot, Cache, and Graph-based logits.
    """
    device = next(clip_model.parameters()).device
    dtype = next(clip_model.parameters()).dtype
    image_model.to(device)

    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(device).to(dtype)
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(image_model.parameters()),
        lr=wandb.config.lr,
        eps=1e-4,
        weight_decay=wandb.config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    # Get all three hyperparameters
    beta, alpha = wandb.config.init_beta, wandb.config.init_alpha
    gamma = wandb.config.init_gamma # <<< NEW: Get gamma from config
    lambda_con = wandb.config.lambda_con
    # <<< NEW: Get the gamma hyperparameter for Focal Loss >>>
    focal_gamma = wandb.config.focal_loss_gamma
    
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        adapter.train()
        image_model.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print(f'Train Epoch: {train_idx} / {cfg["train_epoch"]}')

        for i, (batched_graphs, images, target) in enumerate(tqdm(train_loader_F)):
            batched_graphs, images, target = batched_graphs.to(device), images.to(device), target.to(device)
            
            # --- Get all three feature types ---
            
            # 1. Get graph-refined features from your HGT model
            graph_visual_feature, all_updated_text_features = image_model(
                batched_graphs.x_dict,
                batched_graphs.edge_index_dict,
                batched_graphs.batch_dict
            )
            
            # 2. Get standard global features for zero-shot and cache calculation
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # --- Calculate all three logit components ---

            # 1. Zero-Shot Logits (Original CLIP)
            # We use the standard `image_features` and the pre-computed `clip_weights`.
            original_clip_logits = 100. * image_features.to(dtype) @ clip_weights.to(dtype)

            # 2. Cache Logits (Tip-Adapter)
            # This also uses the standard `image_features`.
            affinity = adapter(image_features.to(dtype))
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(dtype)
            
            # 3. Graph-Refined Logits (HGT-based)
            # This is the logic you already developed. Let's call it `graph_logits` for clarity.
            graph_visual_feature = graph_visual_feature / graph_visual_feature.norm(dim=-1, keepdim=True)
            all_updated_text_features = all_updated_text_features / all_updated_text_features.norm(dim=-1, keepdim=True)
            B, D, C = graph_visual_feature.shape[0], graph_visual_feature.shape[1], all_updated_text_features.shape[0] // graph_visual_feature.shape[0]
            updated_text_features_per_image = all_updated_text_features.view(B, C, D)
            visual_feature_for_bmm = graph_visual_feature.unsqueeze(1)
            text_features_for_bmm = updated_text_features_per_image.transpose(1, 2)
            graph_logits = 100. * torch.bmm(visual_feature_for_bmm, text_features_for_bmm).squeeze(1)
            
            # --- Combine the logits from the three "experts" ---
            # <<< MODIFIED: This is the core of your new idea!
            tip_logits = original_clip_logits + cache_logits * alpha + graph_logits * gamma

            # --- Calculate Losses ---
            # This is the main classification loss on the combined logits
            loss = F.cross_entropy(tip_logits, target)
            
            # <<< MODIFIED: Implement Focal Loss for the contrastive alignment loss >>>
            # This replaces `loss_con = F.cross_entropy(graph_logits, target)`
            
            # 1. Get the log-probabilities and probabilities from the graph logits
            log_pt = F.log_softmax(graph_logits, dim=1)
            pt = torch.exp(log_pt)

            # 2. Get the probabilities for the *correct* class
            # This uses "advanced indexing" to pick the probability corresponding to the target label
            pt_correct = pt[torch.arange(B, device=device), target]

            # 3. Get the log-probabilities for the *correct* class
            log_pt_correct = log_pt[torch.arange(B, device=device), target]

            # 4. Compute the Focal Loss
            # loss = - (1 - pt_correct)^gamma * log_pt_correct
            # We use the existing variable `loss_con` as requested, but it now holds the Focal Loss
            loss_con = -torch.pow(1 - pt_correct, focal_gamma) * log_pt_correct
            loss_con = loss_con.mean() # Average the loss across the batch
            
            # <<< END OF MODIFICATION >>>
            
            total_loss = loss + lambda_con * loss_con
            
            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            
            loss_list.append(total_loss.item())
            
            if(wandb_log):
                wandb.log({
                    "Train accuracy": correct_samples / all_samples,
                    "Total loss": sum(loss_list)/len(loss_list),
                    "Loss (Classification)": loss.item(),
                    # <<< MODIFIED: Log description to reflect Focal Loss >>>
                    "Loss (Focal Contrastive)": loss_con.item(),
                    "Learning rate": scheduler.get_last_lr()[0]
                })

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        # <<< MODIFIED: Fixed a small typo in your original code (list_list -> loss_list) >>>
        print(f'LR: {current_lr:.6f}, Acc: {correct_samples / all_samples:.4f} ({correct_samples}/{all_samples}), Loss: {sum(loss_list)/len(loss_list):.4f}')

        # --- Evaluation Phase (uses standard test features, remains unchanged) ---
        adapter.eval()
        image_model.eval()
        
        with torch.no_grad():
            affinity = adapter(test_features.to(dtype))
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(dtype)
            
            clip_logits = 100. * test_features.to(dtype) @ clip_weights.to(dtype)
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, test_labels)
        
        if wandb_log:
            # <<< CHANGED: Log test accuracy per epoch
            wandb.log({"Test accuracy": acc, "Epoch": train_idx})

        print(f"**** Tip-Adapter-F's test accuracy: {acc:.2f}. ****\n")
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.state_dict(), os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"))

    adapter.load_state_dict(torch.load(os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"), map_location=device))
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    # <<< CHANGED: Log the best accuracy to wandb.summary for the sweep
    if wandb_log:
        wandb.summary["best_test_accuracy"] = best_acc
    
    print("\n-------- Searching hyperparameters on the val set. --------")
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values.to(dtype), val_features.to(dtype), val_labels, clip_weights.to(dtype), adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
    
    affinity = adapter(test_features.to(dtype))
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values.to(dtype)
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    final_acc = max(best_acc, acc)
    print("**** {} Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(cfg['dataset'], final_acc))
    
    # <<< CHANGED: Log final accuracy to wandb.summary
    if wandb_log:
        wandb.summary["final_test_accuracy_after_search"] = final_acc


def main():
    
    # <<< CHANGED: Added ArgumentParser to be controlled by W&B Sweep
    parser = argparse.ArgumentParser()
    # Config and dataset args
    parser.add_argument('--config', type=str, default='./configs/fgvc.yaml', help='Path to dataset config file')
    parser.add_argument('--shots', type=int, default=4, help='Number of few-shot samples')
    # Optimizer args
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    # Model architecture args
    parser.add_argument('--hgt_num_layers', type=int, default=3, help='Number of HGT layers')
    parser.add_argument('--hgt_num_heads', type=int, default=16, help='Number of HGT attention heads')
    parser.add_argument('--transformer_num_layers', type=int, default=3, help='Number of Transformer Encoder layers')
    parser.add_argument('--transformer_nhead', type=int, default=16, help='Number of attention heads in the transformer encoder')
    parser.add_argument('--transformer_ff_multiplier', type=int, default=2, help='Feed-forward network dimension multiplier')
    parser.add_argument('--pooling_ratio', type=float, default=0.25, help='TopKPooling ratio')
    # Regularization args
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    # Training args
    parser.add_argument('--train_epoch', type=int, default=100, help='Total training epochs')
    parser.add_argument('--init_beta', type=float, default=2.0493246903001525, help='Initial beta for Tip-Adapter')
    parser.add_argument('--init_alpha', type=float, default=9.806773071379816, help='Initial alpha for Tip-Adapter')
    parser.add_argument('--init_gamma', type=float, default=1.0, help='Initial gamma for graph logits contribution')
    parser.add_argument('--lambda_con', type=float, default=0.5, help='Weight for the direct contrastive loss')
    
    # <<< NEW: Add the single hyperparameter for Focal Loss >>>
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma for Focal Loss (controls hardness focus)')


    args = parser.parse_args()

    # <<< CHANGED: Initialize wandb with config from parser
    # wandb.config will be automatically populated by sweep agent
    # <<< MODIFIED: Updated wandb run name to reflect new loss >>>
    wandb.init(config=args) #, name=f'{args.shots}shots_FocalLoss_noRN', project= "oxford_pets_topk"
    # Use wandb.config for all hyperparameters
    config = wandb.config
    
    # <<< CHANGED: Removed hardcoded loops
    cfg = yaml.load(open(config.config, 'r'), Loader=yaml.Loader)
    cfg["shots"] = config.shots
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['train_epoch'] = config.train_epoch # Use train_epoch from sweep

    print("\nRunning configs.")
    print(cfg, "\n")
    print("Sweep Hyperparameters:")
    print(wandb.config)
    
    # ... (Dataset specific num_classes logic remains the same)
    if(cfg['dataset'] == "birds"):
        num_classes = 200
    elif(cfg['dataset'] == "dogs"):
        num_classes = 120
    elif(cfg['dataset'] == "cars"):
        num_classes = 196
    elif(cfg['dataset'] == "oxford_pets"):
        num_classes = 37
    elif(cfg['dataset'] == "flowers"):
        num_classes = 102
    elif(cfg['dataset'] == "food101"):
        num_classes = 101
    elif(cfg['dataset'] == "dtd"):
        num_classes = 47
    elif(cfg['dataset'] == "aircrafts"):
        num_classes = 100
    elif(cfg['dataset'] == "ucf101"):
        num_classes = 101

    # if(wandb_log):
        # <<< CHANGED: wandb.init is already called. Left login key.
        # wandb.login(key="657cad5a7ed6fb8cbcfbc5d725d7ca766e460483") #enter your own wandb key
        # wandb.init( ... ) # This is now done at the top of main()

    # <<< --- START OF METADATA AND MODEL CHANGES --- >>>

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # <<< CHANGED: Define graph metadata with three node types
    node_types = ['vit', 'text']
    edge_types = [
        ('vit', 'intra_patch', 'vit'),
        # ('resnet', 'intra_patch', 'resnet'),
        # ('text', 'intra_text', 'text'),
        # ('vit', 'inter_patch', 'resnet'),
        # ('resnet', 'inter_patch', 'vit'),
        ('vit', 'visual_to_text', 'text'),
        ('text', 'text_to_visual', 'vit'),
        # ('resnet', 'visual_to_text', 'text'),
        # ('text', 'text_to_visual', 'resnet'),
    ]

    # --- Load CLIP Models and prepare initial features ---
    clip_model_vit, vit_preprocess = clip.load(cfg['backbone'][0], device=device)
    clip_model_vit = clip_model_vit.float()
    
    # clip_model_resnet, resnet_preprocess = clip.load(cfg['backbone'][1], device=device)
    # clip_model_resnet = clip_model_resnet.float()
    patch_processor = vit_preprocess

    # <<< ADDED: Prepare datasets first to get classnames for text features
    random.seed(1)
    torch.manual_seed(1)
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    # <<< ADDED: Generate text features *once* to be used for all graphs
    print("\nGetting textual features for graph nodes and CLIP's classifier.")
    # This tensor will serve both as the initial text node features and the final classifier for eval
    text_features_for_nodes = clip_classifier(dataset.classnames, dataset.template, clip_model_vit)
    # Transpose it to be [num_classes, feature_dim] for node features
    text_features_for_nodes = text_features_for_nodes.t()
    
    # For the final evaluation, we need the original [feature_dim, num_classes]
    clip_weights = text_features_for_nodes.t()

    # --- Initialize the HGT Model ---
    # <<< CHANGED: Update input_dims to include the text modality
    input_dims = {
        'vit': clip_model_vit.visual.output_dim,
        # 'resnet': clip_model_resnet.visual.output_dim,
        'text': clip_model_vit.text_projection.shape[1] # Or clip_model_vit.ln_final.weight.shape[0]
    }
    hidden_dim = clip_model_vit.visual.output_dim

    image_model = HGTImageFeatureExtractor(
        node_types=node_types,
        edge_types=edge_types,
        input_dims=input_dims,
        hidden_channels=hidden_dim,
        hgt_num_heads=config.hgt_num_heads,
        hgt_num_layers=config.hgt_num_layers,
        dropout_rate=config.dropout_rate,
        transformer_nhead=config.transformer_nhead,
        transformer_num_layers=config.transformer_num_layers,
        transformer_ff_multiplier=config.transformer_ff_multiplier,
        transformer_activation='gelu',
        pooling_ratio=config.pooling_ratio,
        shots=config.shots
    )
    image_model.to(device) # <<< ADDED: Move model to device

    # --- Prepare Datasets and DataLoaders ---
    # Dataset was already built above, now build loaders
    
    val_loader = build_data_loader(data_source=dataset.val, batch_size=2, is_train=False, tfm=vit_preprocess)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=2, is_train=False, tfm=vit_preprocess)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=config.batch_size, is_train=True, tfm=train_transform)
    
    # <<< CHANGED: Pass text_features_for_nodes to the graph data loader
    train_loader_F = build_graph_data_loader(
        data_source=dataset.train_x,
        batch_size=config.batch_size,
        shuffle=True,
        transform=train_transform,
        vit_model=clip_model_vit.visual,
        # resnet_model=clip_model_resnet.visual,
        text_features=text_features_for_nodes, # Pass the text features here
        processor=patch_processor,
        device=device
    )

    # --- Pre-load features and build initial cache model ---
    # clip_weights already created
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model_vit, train_loader_cache)

    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model_vit, val_loader)

    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model_vit, test_loader)

    # --- Run Initial Zero-Shot and Tip-Adapter ---
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # --- Run Fine-Tuning and Evaluation ---
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model_vit, image_model, train_loader_F)
    
    # ... (Rest of main function is identical)
    print("\nRunning configs.")
    print(cfg, "\n")
    print("Sweep Hyperparameters:")
    print(wandb.config)
    
    if wandb_log:
        wandb.finish()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
