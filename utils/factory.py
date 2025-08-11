from models.fddr_cifar import FDDR_CIFAR_Learner
from models.fddr_imagenet import FDDR_ImageNet_Learner


def get_model(model_name, args):
    name = model_name.lower()
    if name == "fddr_cifar":
        return FDDR_CIFAR_Learner(args)
    elif name == "fddr_imagenet":
        return FDDR_ImageNet_Learner(args)
    else:
        assert 0, "Not Implemented!"
