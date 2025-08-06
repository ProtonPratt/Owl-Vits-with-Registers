# modeling_owlvit_with_registers.py
# coding=utf-8
# Copyright 2024 Google AI, The HuggingFace Team & Project Contributors. All rights reserved.
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
"""PyTorch OWL-ViT model with Registers."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.owlv2.modeling_owlv2 import ( # CHANGE HERE
    # Import necessary base classes and functions from OWLv2
    Owlv2TextTransformer,                 # CHANGE HERE
    Owlv2PreTrainedModel,               # CHANGE HERE
    Owlv2VisionEmbeddings,              # CHANGE HERE
    Owlv2Encoder,                       # CHANGE HERE
    Owlv2Model,                         # CHANGE HERE
    Owlv2ForObjectDetection,            # CHANGE HERE
    Owlv2BoxPredictionHead,             # CHANGE HERE
    Owlv2ClassPredictionHead,           # CHANGE HERE (if used directly, maybe not)
    Owlv2Output,                        # CHANGE HERE
    Owlv2ObjectDetectionOutput,         # CHANGE HERE
    Owlv2ImageGuidedObjectDetectionOutput, # CHANGE HERE
    # Import docstrings if needed or copy/adapt them from OWLV2
    OWLV2_START_DOCSTRING,              # CHANGE HERE
    OWLV2_VISION_INPUTS_DOCSTRING,      # CHANGE HERE
    OWLV2_INPUTS_DOCSTRING,             # CHANGE HERE
    OWLV2_OBJECT_DETECTION_INPUTS_DOCSTRING,  # CHANGE HERE
    OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING, # CHANGE HERE
    # Import helper functions (check if still needed or available in OWLv2)
    box_iou,
    generalized_box_iou,
    center_to_corners_format,
)
from .configuration_owlvit_registers import (
    Owlv2ConfigWithRegisters,
    Owlv2TextConfig,
    Owlv2VisionConfigWithRegisters,
)


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/owlvit-base-patch32" # Keep or update as needed


# --- Modified Vision Transformer with Registers ---
class Owlv2VisionTransformerWithRegisters(nn.Module):
    """
    Modified Owlv2VisionTransformer to include register tokens.
    """
    def __init__(self, config: Owlv2VisionConfigWithRegisters):
        super().__init__()
        self.config = config
        self.num_registers = config.num_registers

        self.embeddings = Owlv2VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = Owlv2Encoder(config) # Use the standard encoder
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize register tokens if needed
        if self.num_registers > 0:
            self.registers = nn.Parameter(torch.randn(1, self.num_registers, config.hidden_size))
            # Optional: Initialize registers using truncated normal distribution
            nn.init.trunc_normal_(self.registers, std=config.initializer_range)
        else:
            self.registers = None

    # @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2VisionConfigWithRegisters)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Args:
            interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
                Whether to interpolate the position embeddings if the input image resolution differs from the training resolution.

        Returns:
             Outputs similar to `Owlv2VisionTransformer`, but `last_hidden_state` excludes register tokens.
             `hidden_states` and `attentions` from the encoder will include registers if `num_registers > 0`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Cast the input to the expected `dtype`
        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)

        # Step 1: Get patch and CLS embeddings (standard ViT embeddings)
        embedding_output = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        batch_size, original_num_tokens, hidden_size = embedding_output.shape # original_num_tokens = 1 (CLS) + num_patches

        # Step 2: Prepend register tokens if they exist
        if self.registers is not None:
            # Expand registers to batch size
            registers_expanded = self.registers.expand(batch_size, -1, hidden_size)
            # Concatenate: CLS, Patches, Registers
            # Note: Original OWL-ViT/CLIP embeddings are [CLS, Patches]
            encoder_input = torch.cat([embedding_output, registers_expanded], dim=1)
        else:
            encoder_input = embedding_output

        # Step 3: Pass through pre-layernorm and encoder
        encoder_input_ln = self.pre_layernorm(encoder_input)
        encoder_outputs = self.encoder(
            inputs_embeds=encoder_input_ln,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Step 4: Get the full output sequence from the encoder
        last_hidden_state_full = encoder_outputs[0]

        # Step 5: Crucially, slice out the register tokens before downstream use
        # Keep only the CLS and Patch tokens
        last_hidden_state = last_hidden_state_full[:, :original_num_tokens, :]

        # Step 6: Apply post-layernorm to the CLS token embedding (standard ViT pooling)
        # Use the sliced last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # Step 7: Prepare outputs
        if not return_dict:
            # Return the *sliced* last_hidden_state for downstream tasks
            # Encoder outputs (hidden_states, attentions) will pertain to the full sequence if registers were used
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            # Return the *sliced* last_hidden_state for downstream tasks
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            # Note: hidden_states and attentions from encoder_outputs are for the full sequence (incl. registers)
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# --- Modified Owlv2Model ---
# @add_start_docstrings(OWLVIT_START_DOCSTRING)
class Owlv2ModelWithRegisters(Owlv2PreTrainedModel):
    """
    OWL-ViT model that uses Owlv2VisionTransformerWithRegisters for image encoding.
    Inherits from Owlv2PreTrainedModel for weight initialization etc.
    """
    config_class = Owlv2ConfigWithRegisters # Use the new config class

    def __init__(self, config: Owlv2ConfigWithRegisters):
        super().__init__(config) # Initialize Owlv2PreTrainedModel

        # Ensure config types are correct
        if not isinstance(config.text_config, Owlv2TextConfig):
            raise TypeError("config.text_config must be Owlv2TextConfig")
        if not isinstance(config.vision_config, Owlv2VisionConfigWithRegisters):
            raise TypeError("config.vision_config must be Owlv2VisionConfigWithRegisters")

        text_config = config.text_config
        vision_config = config.vision_config # This is now Owlv2VisionConfigWithRegisters

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # Use the original text model
        self.text_model = Owlv2TextTransformer(text_config)
        # Use the MODIFIED vision model
        self.vision_model = Owlv2VisionTransformerWithRegisters(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    # get_text_features remains the same as Owlv2Model
    def get_text_features(self, *args, **kwargs):
        # Delegate to the original Owlv2Model implementation logic
        # (or copy it here if direct delegation isn't easy/clean)
        temp_model = Owlv2Model(self.config) # Create temporary original model to access method
        temp_model.text_model = self.text_model # Use our text model
        temp_model.text_projection = self.text_projection # Use our projection
        return temp_model.get_text_features(*args, **kwargs)


    # get_image_features remains the same as Owlv2Model - it just calls the vision_model
    # which now handles registers internally
    def get_image_features(self, *args, **kwargs):
        # Delegate to the original Owlv2Model implementation logic
        # (or copy it here if direct delegation isn't easy/clean)
        temp_model = Owlv2Model(self.config) # Create temporary original model to access method
        temp_model.vision_model = self.vision_model # Use our vision model
        temp_model.visual_projection = self.visual_projection # Use our projection
        return temp_model.get_image_features(*args, **kwargs)

    # # forward method remains the same structure, calling the respective models
    # @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Owlv2Output, config_class=Owlv2ConfigWithRegisters)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Owlv2Output]:

        # Call the original forward logic - it will use self.vision_model and self.text_model
        # which are correctly instantiated (vision_model includes registers)
        temp_model = Owlv2Model(self.config) # Create temporary original model to access method
        temp_model.vision_model = self.vision_model
        temp_model.text_model = self.text_model
        temp_model.visual_projection = self.visual_projection
        temp_model.text_projection = self.text_projection
        temp_model.logit_scale = self.logit_scale

        # Pass relevant arguments - ignore 'return_base_image_embeds' if not present
        forward_args = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "return_loss": return_loss,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "interpolate_pos_encoding": interpolate_pos_encoding,
            "return_dict": return_dict,
        }
        return temp_model.forward(**forward_args)


# --- Modified Owlv2ForObjectDetection ---
class Owlv2ForObjectDetectionWithRegisters(Owlv2ForObjectDetection):
    """
    OWL-ViT model with object detection heads that uses Owlv2ModelWithRegisters.
    """
    config_class = Owlv2ConfigWithRegisters # Use the new config class

    def __init__(self, config: Owlv2ConfigWithRegisters):
        # Call the original __init__ but it will use the new config class
        super().__init__(config)

        # Override self.owlvit to be the model with registers
        self.owlvit = Owlv2ModelWithRegisters(config)

        # Heads and other attributes are initialized correctly by super().__init__
        # The compute_box_bias helper also uses self.config.vision_config, which is now the registers version

    # The forward, image_text_embedder, image_embedder, embed_image_query,
    # class_predictor, box_predictor methods rely on self.owlvit (which now handles registers)
    # or config attributes. They should work correctly without needing overrides,
    # as the register handling is encapsulated within the vision transformer.
    # Re-implementing forward just to update docstrings and return types if needed.

    # @add_start_docstrings_to_model_forward(OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Owlv2ObjectDetectionOutput, config_class=Owlv2ConfigWithRegisters)
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Owlv2ObjectDetectionOutput]:
        # Simply call the original forward method from the parent class.
        # It will use self.owlvit which is correctly instantiated as Owlv2ModelWithRegisters.
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

    # @add_start_docstrings_to_model_forward(OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Owlv2ImageGuidedObjectDetectionOutput, config_class=Owlv2ConfigWithRegisters)
    def image_guided_detection(
        self,
        pixel_values: torch.FloatTensor,
        query_pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Owlv2ImageGuidedObjectDetectionOutput]:
         # Simply call the original method from the parent class.
         # It will use self.owlvit which is correctly instantiated as Owlv2ModelWithRegisters.
        return super().image_guided_detection(
            pixel_values=pixel_values,
            query_pixel_values=query_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )


__all__ = [
    "Owlv2ModelWithRegisters",
    "Owlv2ForObjectDetectionWithRegisters",
    # Keep original pre-trained model base if needed for initialization helpers
    "Owlv2PreTrainedModel",
]