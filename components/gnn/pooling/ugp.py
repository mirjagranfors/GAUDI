from deeplay.module import DeeplayModule

from components.gnn.pooling.scc import SelectClusterConnect 

import torch

class UniformGraphPooling(SelectClusterConnect):
    """
    Pools all the nodes of the graph into a single cluster.

    This class inherits from `SelectClusterConnect`, and the selection module
    explicitly assigns all nodes to the same cluster, effectively pooling them
    into a single representation

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the
        following attributes:
        - x: torch.Tensor of shape (num_nodes, node_features)

    - output: Dict[str, Any] or torch-geometric Data object containing the
        following attributes:
        - x: torch.Tensor of shape (num_clusters, node_features)
        - s: torch.Tensor of shape (num_nodes, num_clusters)

    Examples
    --------
    >>> global_pool = UniformGraphPooling().build()
    >>> inp = {}
    >>> inp["x"] = torch.randn(3, 2)
    >>> inp["batch"] = torch.zeros(3, dtype=int)
    >>> inp["edge_index"] = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> out = global_pool(inp)
    """
    def __init__(
            self,
            ):
        super().__init__(
            select=self.UniformSelect(),
            reduce=None,
            pool_loss=None,
        )

    class UniformSelect(DeeplayModule):
        def forward(self, x):
            return torch.ones((x.shape[0], 1)) 