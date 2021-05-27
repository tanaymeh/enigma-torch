def _trainRoutine(*args, **kwargs):
    """
    Main training function for image classification / regression
    """
    model = args[0]
    train_loader = args[1]
    train_loss_fn = args[2]
    optimizer = args[3]
    scheduler = args[4]
    device = args[5]
    apex_status = args[6]
    
    
    