"""Teachers built off their pretraining resolution (all REAMed teachers run at
224 px; MedSAM is 1024-native, MedSigLIP 448-native).

Tiny randomly initialised HF models are saved to a temp dir so no download is
needed.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")


def _tiny_sam(tmp_path, image_size=64):
    from transformers import SamConfig, SamModel
    vision = dict(image_size=image_size, patch_size=16, hidden_size=32, num_hidden_layers=2,
                  num_attention_heads=2, mlp_dim=64, output_channels=16, window_size=2,
                  global_attn_indexes=[1], num_pos_feats=8)
    cfg = SamConfig(vision_config=vision,
                    prompt_encoder_config=dict(hidden_size=16, image_size=image_size,
                                               patch_size=16, image_embedding_size=image_size // 16),
                    mask_decoder_config=dict(hidden_size=16, num_hidden_layers=1,
                                             num_attention_heads=2, mlp_dim=32,
                                             iou_head_hidden_dim=16))
    path = tmp_path / "sam"
    SamModel(cfg).save_pretrained(path)
    return str(path)


def _tiny_siglip(tmp_path, image_size=64):
    from transformers import SiglipConfig, SiglipModel
    cfg = SiglipConfig(
        vision_config=dict(image_size=image_size, patch_size=16, hidden_size=32,
                           num_hidden_layers=2, num_attention_heads=2, intermediate_size=64),
        text_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, vocab_size=100))
    path = tmp_path / "siglip"
    SiglipModel(cfg).save_pretrained(path)
    return str(path)


def test_medsam_teacher_runs_below_native_size(tmp_path):
    from medlat.alignments.teachers.sam_teacher import MedSAMTeacher
    path = _tiny_sam(tmp_path, image_size=64)
    t = MedSAMTeacher(path, img_size=32)
    assert t.vision_encoder.pos_embed.shape[1:3] == (2, 2)
    assert not t.vision_encoder.pos_embed.requires_grad
    feats = t.extract_features(torch.rand(2, 3, 32, 32))
    assert feats.shape == (2, 4, 16)


def test_medsam_teacher_native_size_unchanged(tmp_path):
    from medlat.alignments.teachers.sam_teacher import MedSAMTeacher
    path = _tiny_sam(tmp_path, image_size=64)
    t = MedSAMTeacher(path, img_size=64)
    assert t.vision_encoder.pos_embed.shape[1:3] == (4, 4)
    assert t.extract_features(torch.rand(1, 3, 64, 64)).shape == (1, 16, 16)


def test_medsam_teacher_rejects_non_multiple_of_patch(tmp_path):
    from medlat.alignments.teachers.sam_teacher import MedSAMTeacher
    with pytest.raises(ValueError):
        MedSAMTeacher(_tiny_sam(tmp_path), img_size=40)


def test_hf_teacher_interpolates_pos_encoding_below_native_size(tmp_path):
    from medlat.alignments.teachers.hf_teacher import HuggingFaceTeacher
    path = _tiny_siglip(tmp_path, image_size=64)
    t = HuggingFaceTeacher(path, img_size=32)
    assert t.extract_features(torch.rand(2, 3, 32, 32)).shape == (2, 4, 32)
    # native size still takes the plain path
    t64 = HuggingFaceTeacher(path, img_size=64)
    assert t64.extract_features(torch.rand(1, 3, 64, 64)).shape == (1, 16, 32)


def test_hf_teacher_dinov2_off_native_size(tmp_path):
    # Dinov2 interpolates on its own and rejects interpolate_pos_encoding;
    # rad_dino's config says 518 while REAMed runs it at 224.
    from transformers import Dinov2Config, Dinov2Model
    from medlat.alignments.teachers.hf_teacher import HuggingFaceTeacher
    path = tmp_path / "dinov2"
    Dinov2Model(Dinov2Config(image_size=64, patch_size=16, hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=64)).save_pretrained(path)
    t = HuggingFaceTeacher(str(path), img_size=32)
    assert t.extract_features(torch.rand(2, 3, 32, 32)).shape == (2, 4, 32)
