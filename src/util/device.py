import torch


def use_device(device_setting) -> torch.device:
    use_cuda = device_setting.cuda and torch.cuda.is_available()
    use_mps = device_setting.mps and torch.backends.mps.is_available()
    if use_cuda:
        num_gpu = torch.cuda.device_count()
        print(f"----- Using cuda count: {num_gpu} -----")
        device = torch.device("cuda:0")
    elif use_mps:
        print("----- Using mps -----")
        print("----- Model Dtype must be float32 -----")
        device = torch.device("mps")
    else:
        print("----- Using cpu -----")
        print("----- Model Dtype must be float32 -----")
        device = torch.device("cpu")
    return device
