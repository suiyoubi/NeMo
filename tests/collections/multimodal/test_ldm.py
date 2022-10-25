from dataclasses import asdict
import pytest
from omegaconf import DictConfig, ListConfig
from nemo.collections.multimodal.models.configs.ldm_config import (
    DDPMDiffusionModelConfig,
    LDMFirstStageConfig,
    LatentDiffusionModelConfig, 
    LDMUnetConfig,
    LDMEncoderConfig,
)
from nemo.collections.multimodal.models.ldm.ddpm import DDPM, LatentDiffusion
from nemo.collections.multimodal.modules.diffusionmodules.openaimodel import UNetModel
from nemo.collections.multimodal.models.ldm.autoencoder import AutoencoderKL

@pytest.mark.unit
def test_ddpm_constructor():
    dummy_num_channels = 6
    test_config = DDPMDiffusionModelConfig(channels=dummy_num_channels)
    ddpm = DDPM(cfg=test_config, trainer=None)
    assert isinstance(ddpm, DDPM)
    assert ddpm.channels == dummy_num_channels

    ddpm = DDPM.from_config_dict(test_config)
    assert isinstance(ddpm, DDPM)
    assert ddpm.channels == dummy_num_channels
    print(ddpm._cfg)

def test_ldm_constructor():
    from transformers import logging
    logging.set_verbosity_error()
    dummy_num_channels = 6
    test_config = LatentDiffusionModelConfig(channels=dummy_num_channels)
    ldm = LatentDiffusion(cfg=test_config, trainer=None)
    assert isinstance(ldm, LatentDiffusion)
    assert ldm.channels == dummy_num_channels

    ldm = LatentDiffusion.from_config_dict(test_config)
    assert isinstance(ldm, LatentDiffusion)
    assert ldm.channels == dummy_num_channels

def test_ldm_scheduler_init():
    dummy_num_channels = 6
    test_config = LatentDiffusionModelConfig(channels=dummy_num_channels)
    ldm = LatentDiffusion(cfg=test_config, trainer=None)
    assert ldm.use_scheduler == True
    opt, scheduler = ldm.configure_optimizers()
    assert scheduler is not None
