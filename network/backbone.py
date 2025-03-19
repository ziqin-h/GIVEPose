import copy
import logging
import timm
import pathlib

_logger = logging.getLogger(__name__)


def my_create_timm_model(**init_args):
    # HACK: fix the bug for feature_only=True and checkpoint_path != ""
    # https://github.com/rwightman/pytorch-image-models/issues/488
    if init_args.get("checkpoint_path", "") != "" and init_args.get("features_only", True):
        init_args = copy.deepcopy(init_args)
        full_model_name = init_args["model_name"]
        modules = timm.models.list_modules()
        # find the mod which has the longest common name in model_name
        mod_len = 0
        for m in modules:
            if m in full_model_name:
                cur_mod_len = len(m)
                if cur_mod_len > mod_len:
                    mod = m
                    mod_len = cur_mod_len
        if mod_len >= 1:
            if hasattr(timm.models.__dict__[mod], "default_cfgs"):
                ckpt_path = init_args.pop("checkpoint_path")
                ckpt_url = pathlib.Path(ckpt_path).resolve().as_uri()
                _logger.warning(f"hacking model pretrained url to {ckpt_url}")
                timm.models.__dict__[mod].default_cfgs[full_model_name]["url"] = ckpt_url
                init_args["pretrained"] = True
        else:
            raise ValueError(f"model_name {full_model_name} has no module in timm")
    backbone = timm.create_model(**init_args)
    return backbone

def convnext_backbone():
    INIT_CFG=dict(
                    model_name="convnext_base",
                    pretrained=True,
                    in_chans=3,
                    features_only=True,
                    out_indices=(3,),
                )
    pretrained_cfg = timm.create_model("convnext_base").default_cfg
    INIT_CFG['pretrained_cfg'] = pretrained_cfg # model path
    return my_create_timm_model(**INIT_CFG)

def list_model():
    models_convnext_swin = timm.list_models(pretrained=True)
    # models_convnext_swin = [name for name in models_convnext_swin if
    #                     'convnext_base' in name or 'swinv2' in name]
    print(models_convnext_swin)
    print(len(models_convnext_swin))
    return

if __name__ == '__main__':
    # backbone = convnext_backbone()
    # backbone = swinv2_backbone()
    backbone = list_model()
    # import ipdb
    # ipdb.set_trace()
    # a=1