from typing import Optional, Sequence

from deeplay.module import DeeplayModule
from deeplay.components.mlp import MultiLayerPerceptron

from components.gnn.pooling.scc import SelectClusterConnect 

import torch
import torch.nn as nn

class MinCutPooling(SelectClusterConnect):
    """
    MinCut graph pooling as described in 'Spectral Clustering with Graph Neural
    Networks for Graph Pooling'. This is a graph pooling with a learnable
    selection module.

    Parameters
    ----------
    num_clusters: int
        The number of clusters to which each graph is pooled.
    hidden_features: Sequence[int]
        The number of hidden features for the Multi-Layer Perceptron (MLP) used
        for selecting clusters for the pooling.
    threshold: Optional[float]
        A threshold value to apply to the adjacency matrix to binarize the
        edges. If None, no threshold is applied.
 
    Configurables
    -------------
    - num_clusters (int): The number of clusters to which each graph is pooled.
    - hidden_features (list[int]): The number of hidden features for the
        Multi-Layer Perceptron (MLP) used for selecting clusters for the
        pooling.
    - threshold (float): A threshold value to apply to the adjacency matrix to
        binarize the edges. If None, no threshold is applied. Default is None.

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the
        following attributes:
        - x: torch.Tensor of shape (num_nodes, node_features)
        - edge_index: torch.Tensor of shape (2, num_edges)
        - batch: torch.Tensor of shape (num_nodes)

    """
    def __init__(
            self,
            num_clusters: int,
            hidden_features: Sequence[int],
            threshold: Optional[float] = None,
            ):
        super().__init__(
            select=MultiLayerPerceptron(
                in_features=None,
                hidden_features=hidden_features,
                out_features=num_clusters,
                out_activation=nn.Softmax(dim=1)),
            pool_loss=self.MinCutLoss(),
            reduce=self.ReduceSelfConnection(), 
            threshold=threshold
        )

    class MinCutLoss(DeeplayModule):
        def __init__(
                self,
                eps: Optional[float] = 1e-15,
            ):
            super().__init__()
            self.eps = eps

        def forward(self, A, S):
            n_nodes = S.size(0)
            n_clusters = S.size(1)         

            if A.is_sparse:
                degree = torch.sum(A, dim=0)
            elif (not A.is_sparse) & (A.size(0) == 2):
                A = torch.sparse_coo_tensor(
                    A,
                    torch.ones(A.size(1)),
                    (n_nodes,) * 2,      
                    device=A.device,
                )
                degree = torch.sum(A, dim=0)
            elif (not A.is_sparse) & len({A.size(0), A.size(1)}) == 1:
                degree = torch.sum(A, dim=0)
            else:
                raise ValueError(
                    "Unsupported adjacency matrix format.",
                    "Ensure it is a pytorch sparse tensor, an edge index "
                    "tensor, or a square dense tensor.",
                    "Consider updating the propagate layer to handle "
                    "alternative formats.",
                ) 

            eps = torch.sparse_coo_tensor(
                indices=torch.arange(n_nodes).repeat(2, 1),
                values=torch.zeros(n_nodes) + self.eps,
                size=(n_nodes, n_nodes),
            )  

            D = torch.eye(n_nodes) * degree + eps

            # cut loss:
            L_cut = - torch.trace(
                torch.matmul(S.transpose(-2,-1), torch.matmul(A, S))
            ) / (torch.trace(
                torch.matmul(S.transpose(-2,-1), torch.matmul(D, S)))
            )

            # orthogonality loss:
            L_ortho = torch.linalg.norm(
                (torch.matmul(S.transpose(-2,-1), S) / torch.linalg.norm(
                    torch.matmul(S.transpose(-2,-1), S), ord = 'fro'
                )) - 
                (torch.eye(n_clusters) / torch.sqrt(torch.tensor(n_clusters))),
                ord = 'fro'
            )

            return L_cut, L_ortho
        
    class ReduceSelfConnection(DeeplayModule):
        def __init__(
                self,
                eps: Optional[float] = 1e-15,
            ):
            super().__init__()
            self.eps = eps

        def forward(self, A):        
            ind = torch.arange(A.shape[0])
            A[ind, ind] = 0                         
            D = torch.einsum('jk->j', A)            
            D_inv_sq = torch.pow(D, -0.5)
            D_inv_sq = torch.where(
                torch.isinf(D_inv_sq),
                torch.tensor(0.0),
                D_inv_sq
            )
            D_inv_sq = torch.diag(D_inv_sq)

            A = D_inv_sq @ A @ D_inv_sq
            return A