from .base_mrl import MatryoshkaContrastiveLoss
from .flag_mrl import FlagMRLLoss

criterion_dict = {"mrl": MatryoshkaContrastiveLoss, "flag_mrl": FlagMRLLoss}


def build_criterion(args):
    if args.kd_loss_type not in criterion_dict:
        raise ValueError(f"Unknown kd_loss_type: {args.kd_loss_type}; choose {list(criterion_dict)}")
    return criterion_dict[args.kd_loss_type](args)
