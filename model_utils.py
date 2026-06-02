import os.path
from collections import OrderedDict

import timm
import torch
from transformers import ViTForImageClassification, ViTConfig

from model_zoo.utils import compare_model_outputs
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.models.layers.head import Head
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from model_zoo.utils import convert_weights

model_dict = {
    'ViT-B_16-in21k':'vit_base_patch16_224',
    'ViT-B_16-clip-openai': 'open-clip:timm/vit_base_patch16_clip_224.openai',
    'ViT-B_16-clip-openai-0.03': 'open-clip:timm/vit_base_patch16_clip_224.openai',
    'ViT-B_16-clip-laion2b': 'open-clip:timm/vit_base_patch16_clip_224.laion400m_e31',
    'ViT-B_16-clip-laion2b-0.03': 'open-clip:timm/vit_base_patch16_clip_224.laion400m_e31',
    'ViT-B_16-mae': 'vit_base_patch16_224',
    'ViT-B_16-dinov2': 'facebook/dino-vitb16',
    'ViT-B_16-in1k': 'vit_base_patch16_224',
    'ViT-B_16-in21k-lp':'vit_base_patch16_224',
    'ViT-B_16-clip-openai-lp': 'openai/clip-vit-base-patch16',
    'ViT-B_16-clip-laion2b-lp': 'open-clip:timm/vit_base_patch16_clip_224.laion400m_e31',
    'ViT-B_16-mae-lp': 'vit_base_patch16_224',
    'ViT-B_16-dinov2-lp': 'facebook/dino-vitb16',
    'ViT-B_16-in1k-lp': 'vit_base_patch16_224',
    'ViT-B_16-scratch': '',
}

ckpt_dict = {
    'ViT-B_16-in21k':'/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_08:05:38/waterbirds/ViT/ViT-B_16-in21k_400.bin',
    'ViT-B_16-clip-openai': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_08:52:23/waterbirds/ViT/clip-openai_700.bin',
    'ViT-B_16-clip-laion2b': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_13:44:12/waterbirds/ViT/clip-laion2b_500.bin',
    'ViT-B_16-mae': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_06:32:22/waterbirds/ViT/mae_300.bin',
    'ViT-B_16-dinov2': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_12:00:34/waterbirds/ViT/dinov2_1200.bin',
    'ViT-B_16-in1k': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_11:13:33/waterbirds/ViT/ViT-B_16-in1k_1100.bin',
    'ViT-B_16-in21k-lp':'/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_21:16:45/waterbirds/ViT/ViT-B_16-in21k_600.bin',
    'ViT-B_16-clip-openai-lp': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_21:34:42/waterbirds/ViT/clip-openai_600.bin',
    'ViT-B_16-clip-laion2b-lp': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_22:28:28/waterbirds/ViT/clip-laion2b_600.bin',
    'ViT-B_16-mae-lp': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-06-04_05:47:10/waterbirds/ViT/mae_600.bin',
    'ViT-B_16-dinov2-lp': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_22:09:52/waterbirds/ViT/dinov2_800.bin',
    'ViT-B_16-in1k-lp': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_21:52:12/waterbirds/ViT/ViT-B_16-in1k_400.bin',
    'ViT-B_16-scratch': '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-06-04_00:44:00/waterbirds/ViT/scratch_400.bin',
}

def convert_hf_to_hookedvit(hf_sd, config):
    vit_sd = OrderedDict()
    num_heads = config.n_heads
    head_dim = config.d_head

    for k, v in hf_sd.items():
        if k == "vit.embeddings.cls_token":
            vit_sd["cls_token"] = v
        elif k == "vit.embeddings.position_embeddings":
            vit_sd["pos_embed.W_pos"] = v.squeeze(0)
        elif k == "vit.embeddings.patch_embeddings.projection.weight":
            vit_sd["embed.proj.weight"] = v
        elif k == "vit.embeddings.patch_embeddings.projection.bias":
            vit_sd["embed.proj.bias"] = v

        elif k.startswith("vit.encoder.layer."):
            parts = k.split(".")
            layer = int(parts[3])
            sub = ".".join(parts[4:])

            prefix = f"blocks.{layer}"

            if sub == "layernorm_before.weight":
                vit_sd[f"{prefix}.ln1.w"] = v
            elif sub == "layernorm_before.bias":
                vit_sd[f"{prefix}.ln1.b"] = v
            elif sub == "layernorm_after.weight":
                vit_sd[f"{prefix}.ln2.w"] = v
            elif sub == "layernorm_after.bias":
                vit_sd[f"{prefix}.ln2.b"] = v

            elif sub == "attention.attention.query.weight":
                vit_sd[f"{prefix}.attn.W_Q"] = v.T.reshape(head_dim * num_heads, num_heads, head_dim).permute(1, 0, 2).contiguous()
            elif sub == "attention.attention.key.weight":
                vit_sd[f"{prefix}.attn.W_K"] = v.T.reshape(head_dim * num_heads, num_heads, head_dim).permute(1, 0, 2).contiguous()
            elif sub == "attention.attention.value.weight":
                vit_sd[f"{prefix}.attn.W_V"] = v.T.reshape(head_dim * num_heads, num_heads, head_dim).permute(1, 0, 2).contiguous()
            elif sub == "attention.output.dense.weight":
                vit_sd[f"{prefix}.attn.W_O"] = v.T.reshape(num_heads, head_dim, head_dim * num_heads).contiguous()

            elif sub == "attention.attention.query.bias":
                vit_sd[f"{prefix}.attn.b_Q"] = v.view(num_heads, head_dim)
            elif sub == "attention.attention.key.bias":
                vit_sd[f"{prefix}.attn.b_K"] = v.view(num_heads, head_dim)
            elif sub == "attention.attention.value.bias":
                vit_sd[f"{prefix}.attn.b_V"] = v.view(num_heads, head_dim)
            elif sub == "attention.output.dense.bias":
                vit_sd[f"{prefix}.attn.b_O"] = v

            elif sub == "intermediate.dense.weight":
                vit_sd[f"{prefix}.mlp.W_in"] = v.T  # Transpose!
            elif sub == "intermediate.dense.bias":
                vit_sd[f"{prefix}.mlp.b_in"] = v
            elif sub == "output.dense.weight":
                vit_sd[f"{prefix}.mlp.W_out"] = v.T  # Transpose!
            elif sub == "output.dense.bias":
                vit_sd[f"{prefix}.mlp.b_out"] = v

        elif k == "vit.layernorm.weight":
            vit_sd["ln_final.w"] = v
        elif k == "vit.layernorm.bias":
            vit_sd["ln_final.b"] = v
        elif k == "classifier.weight":
            vit_sd["head.W_H"] = v.T  # Transpose!
        elif k == "classifier.bias":
            vit_sd["head.b_H"] = v
    return vit_sd

def get_model_from_ckpt(model_name, dataset_name, ckpt_path, ckpt_root_path, device='cuda:0'):
    sd_name = model_dict[model_name]
    # total_ckpt_path = os.path.join(ckpt_root_path, ckpt_path)
    total_ckpt_path = ckpt_path
    state_dict = torch.load(total_ckpt_path, map_location=device)
    model = HookedViT.from_pretrained(
        sd_name,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        allow_failing=True,
    )
    model.cfg.device = device
    model.cfg.normalize_output = False
    setattr(model.cfg, "use_normalization_before_and_after", False)
    config = model.cfg
    config.n_classes = 7
    model.head = Head(config)
    converted_weights = convert_weights(state_dict, model_name, config)
    missing, unexpected = model.load_state_dict(converted_weights, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    # original_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True,
    #                                    num_classes=2,
    #                                    drop_rate=0.1,
    #                                    img_size=224)
    # original_model.load_state_dict(state_dict)
    # compare_model_outputs(model, original_model)
    model = model.to(device)
    return model

def get_model_from_name(model_name, device='cuda:0'):
    sd_name = model_dict[model_name]
    model = HookedViT.from_pretrained(
        sd_name,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        allow_failing=True,
    )
    model.cfg.device = device
    model.cfg.normalize_output = False
    setattr(model.cfg, "use_normalization_before_and_after", False)
    # original_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True,
    #                                    num_classes=2,
    #                                    drop_rate=0.1,
    #                                    img_size=224)
    # original_model.load_state_dict(state_dict)
    # compare_model_outputs(model, original_model)
    model = model.to(device)
    return model

def get_model_from_name_lp(model_name, n_classes, device='cuda:0'):
    sd_name = model_dict[model_name]

    if sd_name == '':
        # Scratch init: use timm's trunc_normal_(std=0.02) init (standard ViT),
        # then convert to HookedViT format via the same pipeline as from_pretrained.
        from vit_prisma.models.model_loader import load_config, ModelType, fill_missing_keys
        from vit_prisma.models.weight_conversion import convert_timm_weights
        ref_name = 'vit_base_patch16_224'
        config = load_config(ref_name, ModelType.VISION)
        config.device = device
        config.normalize_output = False
        setattr(config, "use_normalization_before_and_after", False)
        config.n_classes = n_classes
        # Build HookedViT shell (weights will be overwritten below)
        model = HookedViT(config)
        model.head = Head(config, do_init=True)
        # Init timm ViT with pretrained=False → trunc_normal_(std=0.02) for all linears
        # num_classes must match n_classes so head.W_H shape is consistent after conversion
        timm_model = timm.create_model(ref_name, pretrained=False, num_classes=n_classes)
        timm_sd = timm_model.state_dict()
        converted = convert_timm_weights(timm_sd, config)
        full_sd = fill_missing_keys(model, converted)
        model.load_and_process_state_dict(
            full_sd,
            fold_ln=False,
            center_writing_weights=False,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
        )
        model = model.to(device)
        return model

    model = HookedViT.from_pretrained(
        sd_name,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        allow_failing=True,
    )
    model.cfg.device = device
    model.cfg.normalize_output = False
    setattr(model.cfg, "use_normalization_before_and_after", False)
    config = model.cfg
    config.n_classes = n_classes
    model.head = Head(config, do_init=True)
    model = model.to(device)
    return model

def get_model(model_name, dataset_name, device='cuda'):
    if 'ViT-B_16' in model_name:
        # Create HookedViT
        if 'waterbirds' in dataset_name:
            if model_name.split('_lamb')[0] in model_dict.keys():
                model_name = model_name.split('_lamb')[0]
                sd_name = model_dict[model_name]
                ckpt_path = ckpt_dict[model_name]
                state_dict = torch.load(ckpt_path)
                model = HookedViT.from_pretrained(
                    sd_name,
                    center_writing_weights=False,
                    fold_ln=False,
                    refactor_factored_attn_matrices=False,
                    allow_failing=True,
                )
                model.cfg.device = 'cuda'
                model.cfg.normalize_output = False
                setattr(model.cfg, "use_normalization_before_and_after", False)
                config = model.cfg
                config.n_classes = 2
                model.head = Head(config)
                converted_weights = convert_weights(state_dict, model_name, config)
                missing, unexpected = model.load_state_dict(converted_weights, strict=False)
                print("Missing keys:", missing)
                print("Unexpected keys:", unexpected)
                # original_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True,
                #                                    num_classes=2,
                #                                    drop_rate=0.1,
                #                                    img_size=224)
                # original_model.load_state_dict(state_dict)
                # compare_model_outputs(model, original_model)
                model = model.to(device)
            else:
                hf_sd = torch.load(
                    f'/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_group03/waterbirds/ViT/ViT-B_16final.bin')
                model = HookedViT.from_pretrained(
                    "vit_base_patch16_224",
                    center_writing_weights=False,
                    center_unembed=False,
                    fold_ln=False,
                    refactor_factored_attn_matrices=False,
                )
                model.cfg.device = 'cuda'
                setattr(model.cfg, "use_normalization_before_and_after", False)
                config = model.cfg
                config.n_classes = 2
                model.head = Head(config)

                vit_sd = convert_hf_to_hookedvit(hf_sd, config)

                # 4. Load into HookedViT
                missing, unexpected = model.load_state_dict(vit_sd, strict=False)
                print("Missing keys:", missing)
                print("Unexpected keys:", unexpected)
                model = model.to(device)
        elif 'colored-object' in dataset_name:
            vit_config = HookedViTConfig(
                image_size=224,  # same
                patch_size=16,  # same
                n_layers=12,  # num_hidden_layers
                n_heads=12,  # num_attention_heads
                d_mlp=3072,  # intermediate_size
                d_model=768,  # typically inferred: d_mlp // 3
                d_head=64,  # d_model // n_heads
                n_channels=3,  # num_channels
                n_classes=10,  # num_labels
                use_cls_token=True,  # ViT uses class token by default
                classification_type='cls',
                return_type='class_logits',
                device='cuda',
            )
            if 'scratch' in model_name:
                if 'ERM' in model_name:
                    hf_sd = torch.load(
                        f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/ColoredObject_ERM_ratio1.0_pretrainscratch_seed21_finetunefull-finetune/epoch82_train_92.854_val_71.844_test_22.973.pt')
                elif 'IRM' in model_name:
                    hf_sd = torch.load(
                        f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/ColoredObject_IRM_ratio1.0_pretrainscratch_seed21_finetunefull-finetune/epoch32_train1_49.741_train2_42.168_val_45.378_test_13.821.pt')
                elif 'DRO' in model_name:
                    hf_sd = torch.load(
                        f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/ColoredObject_DRO_ratio1.0_pretrainscratch_seed21_finetunefull-finetune/epoch49_train1_98.916_train2_98.354_val_71.046_test_21.928.pt')
            elif 'IN-21k' in model_name:
                if 'ERM' in model_name:
                    hf_sd = torch.load(
                        f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/ColoredObject_ERM_ratio1.0_pretrainIN-21k_seed21_finetunefull-finetune/epoch16_train_99.023_val_93.477_test_81.733.pt')
                elif 'IRM' in model_name:
                    hf_sd = torch.load(
                        f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/ColoredObject_IRM_ratio1.0_pretrainIN-21k_seed21_finetunefull-finetune/epoch69_train1_80.363_train2_76.411_val_77.382_test_58.199.pt')
                elif 'DRO' in model_name:
                    hf_sd = torch.load(
                        f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/ColoredObject_DRO_ratio1.0_pretrainIN-21k_seed21_finetunefull-finetune/epoch14_train1_97.430_train2_99.365_val_93.196_test_81.883.pt')
            model = HookedViT(vit_config)
            model.cfg.device = 'cuda'
            setattr(model.cfg, "use_normalization_before_and_after", False)
            config = model.cfg

            vit_sd = convert_hf_to_hookedvit(hf_sd, config)

            # 4. Load into HookedViT
            missing, unexpected = model.load_state_dict(vit_sd, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            model = model.to(device)
        elif 'mnist' in dataset_name:
            vit_config = HookedViTConfig(
                image_size=28,  # same
                patch_size=7,  # same
                n_layers=12,  # num_hidden_layers
                n_heads=12,  # num_attention_heads
                d_mlp=3072,  # intermediate_size
                d_model=768,  # typically inferred: d_mlp // 3
                d_head=64,  # d_model // n_heads
                n_channels=3,  # num_channels
                n_classes=2,  # num_labels
                use_cls_token=True,  # ViT uses class token by default
                classification_type='cls',
                return_type='class_logits',
                device='cuda',
            )
            if 'erm' in model_name:
                hf_sd = torch.load(
                    f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/vit_erm/ViT_coloredmnist_erm_test_10.45.pt')
            elif 'irm' in model_name:
                hf_sd = torch.load(
                    f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/vit_irm/ViT_coloredmnist_irm_test_76.835.pt')
            elif 'dro' in model_name:
                hf_sd = torch.load(
                    f'/home/yxpengcs/PycharmProjects/vMIB-circuit/model_zoo/ckpts/vit_dro/ViT_coloredmnist_dro_test_90.900.pt')
            model = HookedViT(vit_config)
            model.cfg.device = 'cuda'
            setattr(model.cfg, "use_normalization_before_and_after", False)
            config = model.cfg

            vit_sd = convert_hf_to_hookedvit(hf_sd, config)

            # 4. Load into HookedViT
            missing, unexpected = model.load_state_dict(vit_sd, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            model = model.to(device)
    elif 'google' in model_name:
        model = HookedViT.from_pretrained(
            "vit_base_patch16_224",
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
        )
        model.cfg.device = 'cuda'
        setattr(model.cfg, "use_normalization_before_and_after", False)
        model = model.to(device)
    elif 'small-ViT' in model_name:
        vit_ckpt = f'/home/yxpengcs/PycharmProjects/vision-grokking/checkpoints/vit_erm/ViT_coloredmnist_erm_test_20.465.pt'
        hf_sd = torch.load(vit_ckpt)
        vit_config = HookedViTConfig(
            image_size=28,  # same
            patch_size=7,  # same
            n_layers=2,  # num_hidden_layers
            n_heads=4,  # num_attention_heads
            d_mlp=256 * 3,  # intermediate_size
            d_model=768,  # typically inferred: d_mlp // 3
            d_head=768 // 4,  # d_model // n_heads
            n_channels=3,  # num_channels
            n_classes=1,  # num_labels
            use_cls_token=True,  # ViT uses class token by default
            classification_type='cls',
            return_type='class_logits',
            device='cuda',
        )
        vit_sd = convert_hf_to_hookedvit(hf_sd, vit_config)
        model = HookedViT(vit_config)
        setattr(model.cfg, "use_normalization_before_and_after", False)
        missing, unexpected = model.load_state_dict(vit_sd, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        model = model.to(device)
    else:
        raise NotImplementedError
    return model