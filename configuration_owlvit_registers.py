# configuration_owlvit_registers.py
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team & Project Contributors. All rights reserved.
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
"""OWL-ViT model configuration with Registers"""

from transformers.models.owlvit.configuration_owlvit import (
    OwlViTTextConfig,  # Use original text config
    OwlViTVisionConfig as OriginalOwlViTVisionConfig, # Rename original vision config
    OwlViTConfig as OriginalOwlViTConfig, # Rename original main config
)
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from typing import Dict

logger = logging.get_logger(__name__)


class OwlViTVisionConfigWithRegisters(OriginalOwlViTVisionConfig):
    r"""
    This is the configuration class to store the configuration of an [`OwlViTVisionModelWithRegisters`].
    It inherits from [`OwlViTVisionConfig`] and adds parameters for register tokens.

    Args:
        num_registers (`int`, *optional*, defaults to 0):
            Number of register tokens to add to the Vision Transformer sequence. If 0, behaves like the standard ViT.
        **kwargs:
            Inherited arguments from [`OwlViTVisionConfig`].
    """

    model_type = "owlvit_vision_model_with_registers" # Different model type to distinguish

    def __init__(
        self,
        num_registers=0, # Default to 0 for backward compatibility if not specified
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_registers = num_registers


class OwlViTConfigWithRegisters(OriginalOwlViTConfig):
    r"""
    [`OwlViTConfigWithRegisters`] is the configuration class to store the configuration of an
    [`OwlViTModelWithRegisters`]. It uses [`OwlViTVisionConfigWithRegisters`] for the vision part.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`OwlViTTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`OwlViTVisionConfigWithRegisters`].
        num_registers (`int`, *optional*, defaults to 0):
             Number of register tokens for the vision model. Will be passed to `OwlViTVisionConfigWithRegisters`.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original OWL-ViT
            implementation.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a dictionary. If `False`, returns a tuple.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "owlvit_with_registers" # Different model type
    # Make sure the sub_configs dictionary points to the correct classes
    sub_configs = {"text_config": OwlViTTextConfig, "vision_config": OwlViTVisionConfigWithRegisters}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        num_registers=0, # Add here for easier top-level setting
        projection_dim=512,
        logit_scale_init_value=2.6592,
        return_dict=True,
        **kwargs,
    ):
        # Ensure 'num_registers' is passed to the vision_config dictionary if provided
        if vision_config is not None and 'num_registers' not in vision_config:
            vision_config['num_registers'] = num_registers
        elif vision_config is None:
            vision_config = {'num_registers': num_registers} # Create dict if None
        else:
            # If num_registers is in vision_config, make sure it matches the top-level arg or warn
            if vision_config.get('num_registers', 0) != num_registers:
                 logger.warning(
                     f"Mismatch between top-level num_registers ({num_registers}) and vision_config num_registers "
                     f"({vision_config.get('num_registers')}). Using the value from vision_config."
                 )
                 # Override top-level if vision_config already had it, to be consistent with super().__init__ logic
                 num_registers = vision_config.get('num_registers', 0)


        # Call the OriginalOwlViTConfig init but explicitly handle the config classes
        # We skip the direct super().__init__(...) call because we need to control the vision_config class type
        PretrainedConfig.__init__(self, **kwargs) # Base init

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the OwlViTTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the OwlViTVisionConfigWithRegisters with default values.")

        # Instantiate the correct config classes
        self.text_config = OwlViTTextConfig(**text_config)
        self.vision_config = OwlViTVisionConfigWithRegisters(**vision_config) # Use the new vision config class

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.return_dict = return_dict
        self.initializer_factor = 1.0

    # Keep from_text_vision_configs for compatibility if needed, ensuring it uses the right vision config class
    @classmethod
    def from_text_vision_configs(cls, text_config: Dict, vision_config: Dict, **kwargs):
        r"""
        Instantiate a [`OwlViTConfigWithRegisters`] (or a derived class) from owlvit text model configuration and owlvit vision
        model configuration with registers.

        Returns:
            [`OwlViTConfigWithRegisters`]: An instance of a configuration object
        """
        # Ensure vision_config is processed by the correct class if instantiated from dicts
        vision_config_obj = OwlViTVisionConfigWithRegisters(**vision_config)
        text_config_obj = OwlViTTextConfig(**text_config)

        config_dict = {}
        config_dict["text_config"] = text_config_obj.to_dict()
        config_dict["vision_config"] = vision_config_obj.to_dict()

        return cls.from_dict(config_dict, **kwargs)


__all__ = ["OwlViTConfigWithRegisters", "OwlViTTextConfig", "OwlViTVisionConfigWithRegisters"]