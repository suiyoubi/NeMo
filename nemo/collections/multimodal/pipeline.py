from io import UnsupportedOperation
from typing import Callable, Dict, List, Optional, Union
from nemo.collections.multimodal.models.configs.ldm_config import LatentDiffusionModelConfig
from nemo.collections.multimodal.models.samplers.ddim import DDIMSampler
from nemo.collections.multimodal.models.ldm.ddpm import LatentDiffusion
from nemo.collections.multimodal.models.samplers.plms import PLMSSampler
from omegaconf import OmegaConf
from PIL import Image
import torch
import time
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

class DiffusionPipeline(Callable):
    """
    Abstract class of inference pipeline of diffusion models.

    The goal is to have a unified interface for different diffusion models that varies 
    in model architecture, scheduler, sampler, etc.

    Ultimately, it should provide functional interfaces simlar to other models in NeMo.
    """
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        refresh_cache: bool = False,
        override_config_path: Optional[str] = None,
        map_location: Optional['torch.device'] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Optional['Trainer'] = None,
        save_restore_connector: SaveRestoreConnector = None,
    ):
        # TODO Have a generic way for loading pretrained models for NeMo
        raise NotImplementedError
    
    @classmethod
    def numpy_to_pil(cls, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images
        

class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        model,
        sampler,
        config,
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.model = model.to(device)
        self.sampler = sampler
        self.config = config
        self.downsampling_factor = 8 # TODO infer this from model
        self.z = model.model.diffusion_model.in_channels # TODO find better way to extract this

    @classmethod
    def from_pretrained(cls, ckpt, sampler_type='DDIM'):
        
        # TODO: Ideally, we should have mapping between ckpt and config.
        config = LatentDiffusionModelConfig()
        model = StableDiffusionPipeline.load_model_from_config(config, f"{ckpt}")
        if sampler_type == 'DDIM':
            sampler = DDIMSampler(model)
        elif sampler_type == 'PLMS':
            sampler = PLMSSampler(model)
        else:
            raise ValueError(f'Sampler {sampler_type} is not supported for {cls.__name__}')

        return StableDiffusionPipeline(model, sampler, config)
    
    def load_model_from_config(config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = LatentDiffusion.from_config_dict(config)
        print('finished')
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model
    
    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        width: int = 512,
        height: int = 512,
        inference_steps: int = 50,
        eta: float = 0.,
        unconditional_guidance_scale: float = 7.5,
        num_images_per_promt: Optional[int] = 1,
        output_type: Optional[str] = 'pil',
    ):
        batch_size = num_images_per_promt

        if isinstance(prompts, str):
            prompts = [prompts]

        output = []
        throughput = []

        for prompt in prompts:
            # Generate Unconditional&Conditional Conditioning
            # e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            tic = time.perf_counter()
            tic_total = tic
            if unconditional_guidance_scale != 1.:
                uc = self.model.get_learned_conditioning(batch_size * [""])
            else:
                uc = None
            c = self.model.get_learned_conditioning(batch_size * [prompt])
            toc = time.perf_counter()
            conditioning_time = toc - tic
            shape = [batch_size, height // self.downsampling_factor, width // self.downsampling_factor]
            x_T = torch.randn(
                [batch_size, self.z, height // self.downsampling_factor, width // self.downsampling_factor], 
                device=self.device
            )
            # Sample x_0 from x_T
            tic = time.perf_counter()
            samples, intermediates = self.sampler.sample(
                S=inference_steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc,
                eta=eta,
                x_T=x_T
            )
            toc = time.perf_counter()
            sampling_time = toc - tic
            # Decode latent to image
            tic = time.perf_counter()
            x_samples = self.model.decode_first_stage(samples)
            toc = time.perf_counter()
            decode_time = toc - tic
            # Recenter images from [-1, 1] to [0, 1]
            x_samples_image = torch.clamp((x_samples + 1.) / 2., min=0., max=1.)
            output.append(x_samples_image)
            toc_total = time.perf_counter()
            total_time = toc_total - tic_total
            throughput.append({
                'text-conditioning-time': conditioning_time,
                'sampling-time': sampling_time,
                'decode-time': decode_time,
                'total_time': total_time,
                'sampling-steps': inference_steps,
            })
        
        if output_type == 'torch':
            return torch.cat(output, dim=0)
        
        output_new = []
        for x_samples_image in output:
            # Convert to numpy
            x_samples_image = x_samples_image.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == 'pil':
                x_sample_image = DiffusionPipeline.numpy_to_pil(x_samples_image)
            output_new.append(x_sample_image)
        return output_new, throughput

            
