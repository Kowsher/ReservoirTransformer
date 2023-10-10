# coding=utf-8

""" ReservoirTransformer model configuration"""
""" Author: Md Kowsher"""
from collections import OrderedDict
from typing import Mapping

from transformers import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)




class ReservoirTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ReservoirTModel`]. It is used to
    instantiate a ReservoirTTimeSeries model according to the specified arguments, defining the model architecture. 

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:

        hidden_size (`int`, *optional*, defaults to 16):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 64):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.


        max_sequence_length (`int`, *optional*, defaults to 500):
            The maximum sequence lenght.
        sequence_length (`int`, *optional*, defaults to 12):
            The  sequence lenght of input which is look-back windows to capture the previous history.
        output_size (`int`, *optional*, defaults to None):
            The output dimension of prediction value. In general for mulitvariate-time series, we use all feature to predict.
        re_output_size (`int`, *optional*, defaults to 4):
            The reservoir output dimension. 
        pred_len (`int`, *optional*, defaults to 720):
            The multivaraite horizons to predict. 
    

        num_reservoirs (`int`, *optional*, defaults to 10):
            The reservoirs for ensembelling (group reservoir)
        reservoir_size (`List[int]`, *optional*, defaults to [30, 15, 20, 25, 30, 35, 40, 45, 50, 50]):
            The  reservoir sizes of group reservoir
        spectral_radius (`List[float]`, *optional*, defaults to [0.6, 0.8, 0.55, 0.6, 0.5, 0.4, 0.3, 0.2, 0.81, 0.05]):
            The spectral radius of each reservoir in group reservoir
        sparsity (`List[float]`, *optional*, defaults to [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]):
            The sparsity rate in each reservoir in group reservoir
        leaky (`List[float]`, *optional*, defaults to [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39]):
            The leaky rate in each reservoir in group reservoir         




        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size [mask or non_mask here] of the `token_type_ids` passed when calling [`ReservoirTModel`] .
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        decoder_dropout (`float`, *optional*):
            The dropout ratio for the classification or regression head.
        problem_type ('str', *optional*):
            Type of problem such as 'regression', 'single_label_classification', 'multi_label_classification'


    Examples:

    ```python
    >>> from configuration import ReservoirTConfig

    >>> # Initializing a trnasformer style configuration
    >>> configuration = ReservoirTConfig()

    >>> # Initializing a model (with random weights) from trnasformer style configuration
    >>> model = ReservoirTTimeSeries(config = configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "ReservoirTransformer"

    def __init__(
        self,
        hidden_size=8,
        output_size=None,
        re_output_size=4,
        num_hidden_layers=4,
        pred_len=720,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_sequence_length=500,
        sequence_length=12, 
        type_vocab_size=2,
        num_reservoirs=10, 
        reservoir_size = [30, 15, 20, 25, 30, 35, 40, 45, 50, 50],
        spectral_radius = [0.6, 0.8, 0.55, 0.6, 0.5, 0.4, 0.3, 0.2, 0.81, 0.05],
        sparsity = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15],
        leaky = [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39],    
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        decoder_dropout=None,
        problem_type=None,
        soft_border=8,
        batch=64, 
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        scaling=True,



        #regressor_dropout=None,

        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.decoder_dropout = decoder_dropout
        self.output_size = output_size
        self.re_output_size = re_output_size
        self.pred_len = pred_len
        self.max_sequence_length = max_sequence_length
        self.sequence_length = sequence_length
        self.problem_type = problem_type
        self.num_reservoirs = num_reservoirs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.reservoir_size = reservoir_size
        self.leaky = leaky
        self.soft_border=soft_border
        self.batch=batch
        self.train_size=train_size 
        self.val_size=val_size
        self.test_size=test_size 
        self.scaling=scaling


class ReservoirTOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )