import torch

def logToUser(text: str, end: str='\n'):
    """Print some information on User Console

    Args:
        text (str): Information you want to print
        end (str, optional): String end symbol. Defaults to '\n'.
    """
    print(f"[INFO]: {text}", end=end)
    

def isValidDevice(device: str):
    """Checks if a given device is valid for torch

    Args:
        device (str): Device to check
    """
    try:
        torch.device(device)
        return True
    except:
        return False