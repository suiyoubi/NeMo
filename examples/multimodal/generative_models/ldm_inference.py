from nemo.collections.multimodal.pipeline import StableDiffusionPipeline
import time

metrics = {}
tic = time.perf_counter()
pipeline = StableDiffusionPipeline.from_pretrained(ckpt = '/aot/ckpts/sd-v1-4.ckpt', sampler_type='PLMS')
toc = time.perf_counter()
loading_time = toc - tic

text = [
    'a photograph of an astronaut riding a horse',
    'a men dancing in the sea',
]
num_images_per_promt = 8
images, batch_time = pipeline(prompts = text, output_type='pil', num_images_per_promt=num_images_per_promt, eta=0)

for item in batch_time:
    for key in item:
        if key not in metrics:
            metrics[key] = [item[key]]
        else:
            metrics[key].append(item[key])
# Calculating average time:
ave_metrics = {}
for key in metrics:
    ave_metrics[f'average-{key}'] = sum(metrics[key]) / len(metrics[key])
# Add batch specific config:
metrics['images-per-batch'] = num_images_per_promt
metrics['loading-time'] = loading_time
print(metrics)
print(ave_metrics)
import os
outpath = 'output/ldm_inference_plms'
os.makedirs(outpath, exist_ok=True)
for text_prompt, pils in zip(text, images):
    for idx, image in enumerate(pils):
        image.save(os.path.join(outpath, f'{text_prompt}_{idx}.png'))