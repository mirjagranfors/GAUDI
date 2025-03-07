from typing import Optional, Sequence, Type, Union

from deeplay import DeeplayModule, Layer, LayerList

from deeplay.components.gnn.tpu import TransformPropagateUpdate
from deeplay.components.gnn.gcn import GraphConvolutionalNeuralNetwork
from deeplay.ops import Cat

import torch
import torch.nn as nn


class GraphConvolutionalNeuralNetworkConcat(GraphConvolutionalNeuralNetwork):
    """
    GraphConvolutionalNeuralNetworkConcat is a variant of 
    GraphConvolutionalNeuralNetwork that incorporates an additional 
    concatenation mechanism. Unlike the GraphConvolutionalNeuralNetwork, it 
    does not utilize normalization.

    Parameters
    ----------
    in_features: int
        The number of input features for the graph nodes.
    hidden_features: Sequence[int]
        The number of hidden features for each hidden layer in the network.
    out_features: int
        The number of output features for the final layer.
    out_activation: Optional[Union[Type[nn.Module], nn.Module, None]]
        An optional activation function to apply to the output layer. If None,
        no activation is applied.

    Configurables
    -------------
    - in_features (int): The number of input features for the graph nodes.
    - hidden_features (list[int]): The number of hidden features for each
        hidden layer.
    - out_features (int): The number of output features for the final layer.
    - out_activation (nn.Module or None): The activation function for the
        output layer.

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the
        following attributes:
        - x: torch.Tensor of shape (num_nodes, node_features)
        - edge_index: torch.Tensor of shape (2, num_edges)

    Example
    --------
        >>> GCNConcat = GraphConvolutionalNeuralNetworkConcat(
        >>>     in_features = 16, 
        >>>     hidden_features = [16, 16], 
        >>>     out_features = 16
        >>> ).build()
        >>> inp = {}
        >>> inp["x"] = torch.randn(10, 16)
        >>> inp['batch'] = torch.zeros(10, dtype=int)
        >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
        >>> output = GCNConcat(inp)
    """
    in_features: int
    hidden_features: Sequence[int]
    out_features: int
    out_activation: Optional[Union[Type[nn.Module], nn.Module, None]]

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            out_activation
        )

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        if in_features is None:
            raise ValueError("in_features must be specified")

        if out_features is None:
            raise ValueError("out_features must be specified")

        if in_features <= 0:
            raise ValueError(
                f"in_features must be positive, got {in_features}"
            )

        if out_features <= 0:
            raise ValueError(
                f"out_features must be positive, got {out_features}"
            )

        if any(h <= 0 for h in hidden_features):
            raise ValueError(
                f"all hidden_features must be positive, got {hidden_features}"
            )

        if out_activation is None:
            out_activation = Layer(nn.Identity)
        elif isinstance(out_activation, type) and \
            issubclass(out_activation, nn.Module):
            out_activation = Layer(out_activation)

        class Propagate(DeeplayModule):
            def forward(self, x, A):
                if A.is_sparse:
                    return torch.spmm(A, x)
                elif (not A.is_sparse) & (A.size(0) == 2):
                    A = torch.sparse_coo_tensor(
                        A,
                        torch.ones(A.size(1)),
                        (x.size(0),) * 2,
                        device=A.device,
                    )
                    return torch.spmm(A, x)
                elif (not A.is_sparse) & \
                    len({A.size(0), A.size(1), x.size(0)}) == 1:
                    return A.type(x.dtype) @ x
                else:
                    raise ValueError(
                        "Unsupported adjacency matrix format.",
                        "Ensure it is a pytorch sparse tensor, "
                        "an edge index tensor, or a square dense tensor.",
                        "Consider updating the propagate layer to handle "
                        "alternative formats.",
                    )

        self.blocks = LayerList()

        for i, (c_in, c_out) in enumerate(
            zip(
                [in_features, *hidden_features],
                [*hidden_features, out_features]
            )
        ):
            transform = Layer(nn.Linear, c_in, c_out)
            transform.set_input_map("x")
            transform.set_output_map("x_prime")

            propagate = Layer(Propagate)
            propagate.set_input_map("x_prime", "edge_index")
            propagate.set_output_map("x_prime")

            update = Layer(nn.ReLU) if i < len(self.hidden_features) \
                else out_activation
            update.set_input_map("x_prime")
            update.set_output_map("x_prime")

            block = TransformPropagateUpdate(
                transform=transform,
                propagate=propagate,
                update=update,
                order=["transform", "update", "propagate"]
            )
            self.blocks.append(block)

        self.concat = Cat()
        self.concat.set_input_map('x_prime', 'x')
        self.concat.set_output_map('x')

        self.dense = Layer(nn.Linear, out_features*2, out_features)
        self.dense.set_input_map('x')
        self.dense.set_output_map('x')

        self.activate = Layer(nn.ReLU)
        self.activate.set_input_map('x')
        self.activate.set_output_map('x')

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.concat(x)
        x = self.dense(x)
        x = self.activate(x)

        return x