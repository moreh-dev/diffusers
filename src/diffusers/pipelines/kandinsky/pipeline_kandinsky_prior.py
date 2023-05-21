# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from torchvision import transforms
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ...models import PriorTransformer
from ...pipelines import DiffusionPipeline
from ...schedulers import UnCLIPScheduler
from ...utils import (
    deprecate,
    BaseOutput,
    is_accelerate_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyPipeline, KandinskyPriorPipeline
        >>> import torch

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained("YiYiXu/Kandinsky-prior")
        >>> pipe_prior.to("cuda")

        >>> prompt = "red cat, 4k photo"
        >>> out = pipe_prior(prompt)
        >>> image_emb = out.images
        >>> zero_image_emb = out.zero_embeds

        >>> pipe = KandinskyPipeline.from_pretrained("YiYiXu/Kandinsky")
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     prompt,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=100,
        ... ).images

        >>> image[0].save("cat.png")
        ```
"""

EXAMPLE_INTERPOLATE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyPriorPipeline, KandinskyPipeline
        >>> from diffusers.utils import load_image
        >>> import PIL

        >>> import torch
        >>> from torchvision import transforms

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained("YiYiXu/Kandinsky-prior", torch_dtype=torch.float16)
        >>> pipe_prior.to("cuda")

        >>> img1 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )

        >>> img2 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/starry_night.jpeg"
        ... )

        >>> images_texts = ["a cat", img1, img2]
        >>> weights = [0.3, 0.3, 0.4]
        >>> image_emb, zero_image_emb = pipe_prior.interpolate(images_texts, weights)

        >>> pipe = KandinskyPipeline.from_pretrained("YiYiXu/Kandinsky", torch_dtype=torch.float16)
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     "",
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=150,
        ... ).images[0]

        >>> image.save("starry_cat.png")
        ```
"""


def _convert_image_to_rgb(image):
    return image.convert("RGB")


image_transforms = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)


@dataclass
class KandinskyPriorPipelineOutput(BaseOutput):
    """
    Output class for KandinskyPriorPipeline.

    Args:
        images (`torch.FloatTensor`)
            clip image embeddings for text prompt
        zero_embeds (`List[PIL.Image.Image]` or `np.ndarray`)
            clip image embeddings for unconditional tokens
    """

    images: Union[torch.FloatTensor, np.ndarray]
    zero_embeds: Union[torch.FloatTensor, np.ndarray]


class KandinskyPriorPipeline(DiffusionPipeline):
    """
    Pipeline for generating image prior for Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: CLIPVisionModelWithProjection,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            image_encoder=image_encoder,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_INTERPOLATE_DOC_STRING)
    def interpolate(
        self,
        images_and_prompts: List[Union[str, PIL.Image.Image, torch.FloatTensor]],
        weights: List[float],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        negative_prior_prompt: Optional[str] = None,
        negative_prompt: Union[str] = "",
        guidance_scale: float = 4.0,
        device=None,
    ):
        """
        Function invoked when using the prior pipeline for interpolation.

        Args:
            images_and_prompts (`List[Union[str, PIL.Image.Image, torch.FloatTensor]]`):
                list of prompts and images to guide the image generation.
            weights: (`List[float]`):
                list of weights for each condition in `images_and_prompts`
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            negative_prior_prompt (`str`, *optional*):
                The prompt not to guide the prior diffusion process. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.

        Examples:

        Returns:
            `tuple`
        """

        device = device or self.device

        if len(images_and_prompts) != len(weights):
            raise ValueError(
                f"`images_and_prompts` contains {len(images_and_prompts)} items and `weights` contains {len(weights)} items - they should be lists of same length"
            )

        image_embeddings = []
        for cond, weight in zip(images_and_prompts, weights):
            if isinstance(cond, str):
                image_emb = self.__call__(
                    cond,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    latents=latents,
                    negative_prompt=negative_prior_prompt,
                    guidance_scale=guidance_scale,
                ).images

            elif isinstance(cond, (PIL.Image.Image, torch.Tensor)):
                if isinstance(cond, PIL.Image.Image):
                    cond = image_transforms(cond).unsqueeze(0).to(dtype=self.image_encoder.dtype, device=device)

                image_emb = self.image_encoder(cond)["image_embeds"]

            else:
                raise ValueError(
                    f"`images_and_prompts` can only contains elements to be of type `str`, `PIL.Image.Image` or `torch.Tensor`  but is {type(cond)}"
                )

            image_embeddings.append(image_emb * weight)

        image_emb = torch.cat(image_embeddings).sum(dim=0, keepdim=True)

        out_zero = self.__call__(
            negative_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            negative_prompt=negative_prior_prompt,
            guidance_scale=guidance_scale,
        )
        zero_image_emb = out_zero.zero_embeds if negative_prompt == "" else out_zero.images

        return image_emb, zero_image_emb

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def get_zero_embed(self, batch_size=1, device=None):
        device = device or self.device
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=device, dtype=self.image_encoder.dtype
        )
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
        models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
        when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        models = [
            self.image_encoder,
            self.text_encoder,
        ]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.text_encoder, "_hf_hook"):
            return self.device
        for module in self.text_encoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder(text_input_ids.to(device))

        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 4.0,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generate image. Choose between: `"np"` (`np.array`) or `"pt"`
                (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`KandinskyPriorPipelineOutput`] or `tuple`
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # prior
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        prior_timesteps_tensor = self.scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim

        latents = self.prepare_latents(
            (batch_size, embedding_dim),
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=prompt_embeds,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            ).predicted_image_embedding

            if do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == prior_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = prior_timesteps_tensor[i + 1]

            latents = self.scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

        latents = self.prior.post_process_latents(latents)

        image_embeddings = latents
        zero_embeds = self.get_zero_embed(latents.shape[0], device=latents.device)
 
        if output_type not in [ "pt", "np"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`np`, `pt``"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"

        if output_type == "np":
            image_embeddings = image_embeddings.cpu().numpy()
            zero_embeds = zero_embeds.cpu().numpy()

        if not return_dict:
            return (image_embeddings, zero_embeds)

        return KandinskyPriorPipelineOutput(images=image_embeddings, zero_embeds=zero_embeds)