from deeplay.module import DeeplayModule

import torch
    
class GraphNodeUpsampling(DeeplayModule):
    """
    Upsampling the node features of a graph.

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the
        following attributes:
        - x: torch.Tensor of shape (num_clusters, node_features)
        - s: torch.Tensor of shape (num_nodes, num_clusters)

    - output: Dict[str, Any] or torch-geometric Data object containing the
        following attribute:
        - x: torch.Tensor of shape (num_nodes, node_features)
        
    Examples
    --------
    >>> upsampling = GraphNodeUpsampling()
    >>> upsampling = upsampling.build()

    >>> inp = {}
    >>> inp["x"] = torch.randn(1, 2)
    >>> inp["s"] = torch.ones((3, 1))
    >>> out = upsampling(inp)
    """

    def __init__(
            self,
            ):
        super().__init__()
    
        class Upsample(DeeplayModule):
            def forward(self, x, s):
                return torch.matmul(s, x)
            
        self.upsample = Upsample()
        self.upsample.set_input_map('x', 's')
        self.upsample.set_output_map('x')

    def forward(self, x):
        x = self.upsample(x)
        return x