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

from transformers.models.owlv2.configuration_owlv2 import ( # CHANGE HERE
    Owlv2TextConfig,                             # CHANGE HERE
    Owlv2VisionConfig as OriginalOwlv2VisionConfig, # CHANGE HERE
    Owlv2Config as OriginalOwlv2Config,           # CHANGE HERE
)
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from typing import Dict

logger = logging.get_logger(__name__)


class Owlv2VisionConfigWithRegisters(OriginalOwlv2VisionConfig):
    r"""
    This is the configuration class to store the configuration of an [`Owlv2VisionModelWithRegisters`].
    It inherits from [`Owlv2VisionConfig`] and adds parameters for register tokens.

    Args:
        num_registers (`int`, *optional*, defaults to 0):
            Number of register tokens to add to the Vision Transformer sequence. If 0, behaves like the standard ViT.
        **kwargs:
            Inherited arguments from [`Owlv2VisionConfig`].
    """

    model_type = "owlvit_vision_model_with_registers" # Different model type to distinguish

    def __init__(
        self,
        num_registers=0, # Default to 0 for backward compatibility if not specified
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_registers = num_registers


class Owlv2ConfigWithRegisters(OriginalOwlv2Config):
    r"""
    [`Owlv2ConfigWithRegisters`] is the configuration class to store the configuration of an
    [`Owlv2ModelWithRegisters`]. It uses [`Owlv2VisionConfigWithRegisters`] for the vision part.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Owlv2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Owlv2VisionConfigWithRegisters`].
        num_registers (`int`, *optional*, defaults to 0):
             Number of register tokens for the vision model. Will be passed to `Owlv2VisionConfigWithRegisters`.
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
    sub_configs = {"text_config": Owlv2TextConfig, "vision_config": Owlv2VisionConfigWithRegisters}

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


        # Call the OriginalOwlv2Config init but explicitly handle the config classes
        # We skip the direct super().__init__(...) call because we need to control the vision_config class type
        PretrainedConfig.__init__(self, **kwargs) # Base init

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Owlv2TextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Owlv2VisionConfigWithRegisters with default values.")

        # Instantiate the correct config classes
        self.text_config = Owlv2TextConfig(**text_config)
        self.vision_config = Owlv2VisionConfigWithRegisters(**vision_config) # Use the new vision config class

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.return_dict = return_dict
        self.initializer_factor = 1.0

    # Keep from_text_vision_configs for compatibility if needed, ensuring it uses the right vision config class
    @classmethod
    def from_text_vision_configs(cls, text_config: Dict, vision_config: Dict, **kwargs):
        r"""
        Instantiate a [`Owlv2ConfigWithRegisters`] (or a derived class) from owlvit text model configuration and owlvit vision
        model configuration with registers.

        Returns:
            [`Owlv2ConfigWithRegisters`]: An instance of a configuration object
        """
        # Ensure vision_config is processed by the correct class if instantiated from dicts
        vision_config_obj = Owlv2VisionConfigWithRegisters(**vision_config)
        text_config_obj = Owlv2TextConfig(**text_config)

        config_dict = {}
        config_dict["text_config"] = text_config_obj.to_dict()
        config_dict["vision_config"] = vision_config_obj.to_dict()

        return cls.from_dict(config_dict, **kwargs)


__all__ = ["Owlv2ConfigWithRegisters", "Owlv2TextConfig", "Owlv2VisionConfigWithRegisters"]