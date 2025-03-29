import argparse
#import re
import os
import torch
from data_processing import get_dataloaders
from torch.optim.lr_scheduler import CosineAnnealingLR
#from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import FusionModel
from train import train_model
from utils import *
from config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Fusion Model")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=None)
    parser.add_argument("--lr", type=float, help="Learning rate", default=None)
    args = parser.parse_args()

    # Handle config overrides
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
        print(f"Overriding epochs to {NUM_EPOCHS}")
    
    if args.lr is not None:
        LEARNING_RATE = args.lr
        print(f"Overriding learning rate to {LEARNING_RATE}")

    # Create dataloaders
    print("\nCreating data loaders...")
    train_loader, val_loader = get_dataloaders()
    
    # Initialize model
    print("\nInitializing model...")
    """ model = FusionModel(
        num_classes=NUM_CLASSES,
        num_frames=NUM_FRAMES,
        skeleton_dim=75
    ) """
    
    model = FusionModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    
    # Model verification
    print("\nVerifying model...")
    dummy_rgb = torch.randn(2, 3, NUM_FRAMES, 224, 224)
    dummy_skel = torch.randn(2, NUM_FRAMES, 75)
    try:
        output = model(dummy_rgb, dummy_skel)
        print(f"Model verification passed! Output shape: {output.shape}")
    except Exception as e:
        print(f"Model verification failed: {str(e)}")
        exit(1)

    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Checkpoint handling
    start_epoch = 0
    skeleton_mean = 0
    skeleton_std = 0
    scheduler = None
    
    if os.path.exists(CHECKPOINT_DIR):
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("epoch_")]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            print(f"\nFound checkpoint: {latest_checkpoint}")
            
            checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
            #start_epoch = load_checkpoint(model, checkpoint_path, optimizer)
            start_epoch, skeleton_mean, skeleton_std = load_checkpoint(model, checkpoint_path, optimizer)
            
            # Re-initialize scheduler with remaining epochs
            remaining_epochs = NUM_EPOCHS - start_epoch
            if remaining_epochs > 0:
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)
                #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
                checkpoint = torch.load(checkpoint_path)
                if checkpoint.get('scheduler', None):
                    scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                scheduler = None


    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) if NUM_EPOCHS > 0 else None
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)

    # Start training
    print("\nStarting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        start_epoch=start_epoch,
        optimizer=optimizer,
        scheduler=scheduler
    )