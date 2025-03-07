from typing import Optional, Sequence, Type, Union

from deeplay import Layer, LayerList
from deeplay.ops import Cat

from deeplay.components.gnn.tpu import TransformPropagateUpdate
from deeplay.components.gnn.mpn import MessagePassingNeuralNetwork
from deeplay.components.gnn.mpn import Mean
from deeplay.components.gnn.mpn import Update

from .transformation import TransformSender

import torch.nn as nn

class MessagePassingNeuralNetworkSender(MessagePassingNeuralNetwork):
    hidden_features: Sequence[Optional[int]]
    out_features: int
    blocks: LayerList[TransformPropagateUpdate]

    def __init__(
        self,
        hidden_features: Sequence[int],
        out_features: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__(hidden_features, out_features, out_activation)

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, type) and \
            issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        self.blocks = LayerList()
        for i, c_out in enumerate([*hidden_features, out_features]):
            activation = (
                Layer(nn.ReLU) if i < len(hidden_features) - 1
                else out_activation
            )

            transform = TransformSender(
                combine=Cat(),
                layer=Layer(nn.LazyLinear, c_out),
                activation=activation.new(),
            )
            transform.set_input_map("x", "edge_index", "input_edge_attr")
            transform.set_output_map("edge_attr")

            propagate = Mean()
            propagate.set_input_map("x", "edge_index", "edge_attr")
            propagate.set_output_map("aggregate")

            update = Update(
                combine=Cat(),
                layer=Layer(nn.LazyLinear, c_out),
                activation=activation.new(),
            )
            update.set_input_map("x", "aggregate")
            update.set_output_map("x")

            block = TransformPropagateUpdate(
                transform=transform,
                propagate=propagate,
                update=update,
            )
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
