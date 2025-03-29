import torch
from config import *

""" def validate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
 """
def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model.load_state_dict(checkpoint['model'])
    
    # Load optimizer
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Move optimizer tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # Load scheduler
    if scheduler and checkpoint.get('scheduler', None):
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Load normalized skeleton stats, calculated from training data
    skeleton_mean = checkpoint['skeleton_mean'].cpu().numpy()
    skeleton_std = checkpoint['skeleton_std'].cpu().numpy()
    
    # Return just the epoch number as integer
    return checkpoint['epoch'], skeleton_mean, skeleton_std

""" def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    # Returns metadata about the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Move optimizer tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    scheduler_state = checkpoint.get('scheduler', None)
    scheduler_T_max = checkpoint.get('scheduler_T_max', None)
    scheduler_base_lrs = checkpoint.get('scheduler_base_lrs', None)
    
    if scheduler and scheduler_state:
        scheduler.load_state_dict(scheduler_state)
    
    return {
        'epoch': checkpoint['epoch'],
        'scheduler_state': scheduler_state,
        'scheduler_T_max': scheduler_T_max,
        'scheduler_base_lrs': scheduler_base_lrs
    } """