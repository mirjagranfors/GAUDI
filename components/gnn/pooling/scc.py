from typing import Optional

from deeplay import DeeplayModule

import torch
import torch.nn as nn

class SelectClusterConnect(DeeplayModule):
    """
    A generalized graph pooling module that performs cluster selection,
    pooling of features, and reduction of self-connectivity in the adjacency
    matrix.

    Parameters
    ----------
    select: nn.Module
        A module that selects clusters for pooling.
    pool_loss: Optional[nn.Module], default=None
        A loss function applied during pooling, such as a MinCut loss,
        if specified.
    reduce: Optional[nn.Module], default=None
        A module to reduce the self-connectivity of the adjacency matrix,
        such as a self-connection removal module.
    threshold: Optional[float], default=None
        A threshold value to apply to the adjacency matrix to binarize the
        edges. If None, no threshold is applied.
    
    Configurables
    -------------
    - select (nn.Module): Module used for selecting clusters for pooling.
    - pool_loss (nn.Module, optional): A loss function applied during pooling.
        Default is None.
    - reduce (nn.Module, optional): A module to reduce self-connectivity.
        Default is None.
    - threshold (float, optional): A threshold for binarizing edges in the
        adjacency matrix. Default is None.

    """
    select: nn.Module
    pool_loss: Optional[nn.Module]
    reduce: Optional[nn.Module]
    threshold: Optional[float]

    def __init__(
        self,
        select: nn.Module,
        pool_loss: Optional[nn.Module] = None,
        reduce: Optional[nn.Module] = None,
        threshold: Optional[float] = None,
    ):
        super().__init__()

        class Cluster(DeeplayModule):
            def forward(self, x, s):
                return torch.matmul(s.transpose(-2,-1), x)
            
        class Connect(DeeplayModule):
            def forward(self, A, s):
                if A.is_sparse:
                    return torch.spmm(s.transpose(-2,-1), torch.spmm(A, s))
                elif (not A.is_sparse) & (A.size(0) == 2):
                    A = torch.sparse_coo_tensor(
                        A,
                        torch.ones(A.size(1)),
                        (s.size(0),) * 2,
                        device=A.device,
                    )
                    return torch.spmm(s.transpose(-2,-1), torch.spmm(A, s))
                elif (not A.is_sparse) & (
                    len({A.size(0), A.size(1), s.size(0)}) == 1
                ):
                    return s.transpose(-2,-1) @ A.type(s.dtype) @ s
                else:
                    raise ValueError(
                        "Unsupported adjacency matrix format.",
                        "Ensure it is a pytorch sparse tensor, an edge index "
                        "tensor, or a square dense tensor.",
                        "Consider updating the propagate layer to handle "
                        "alternative formats.",
                    )     
            
        class BatchCompatible(DeeplayModule):
            def forward(self, S, B):
                """Ensures S is compatible with batch-wise processing."""
                unique_graphs = torch.unique(B)
                num_graphs = len(unique_graphs)

                S_ = torch.zeros(S.shape[0] * S.shape[1] * num_graphs)

                row_indices = torch.arange(S.shape[0]).repeat_interleave(
                    S.shape[1]
                )
                col_indices = (
                    B.repeat_interleave(S.shape[1]) * S.shape[1]
                    + torch.arange(S.shape[1]).repeat(S.shape[0])
                )

                S_[row_indices * (S.shape[1] * num_graphs) + col_indices] = (
                    S.view(-1)
                )

                B_ = torch.arange(num_graphs).repeat_interleave(S.shape[1])

                return  S_.reshape([S.shape[0], -1]), B_
            
        class ApplyThreshold(DeeplayModule):
            def __init__(self, threshold: float):
                super().__init__()
                self.threshold = threshold

            def forward(self, A):
                return torch.where(A >= threshold, 1.0, 0.0)

        class SparseEdgeIndex(DeeplayModule):
            """ output edge index as a sparse tensor """
            def forward(self, A):
                if A.is_sparse:
                    edge_index = A
                    return edge_index
                else:
                    edge_index = A.to_sparse()  
                    return edge_index  
            

        self.select = select   
        self.select.set_input_map('x')
        self.select.set_output_map('s')

        self.batch_compatible = BatchCompatible()
        self.batch_compatible.set_input_map('s', 'batch')
        self.batch_compatible.set_output_map('s', 'batch')

        self.pool_loss = pool_loss
        if self.pool_loss is not None:
            self.pool_loss.set_input_map('edge_index', 's')

        self.cluster = Cluster()
        self.cluster.set_input_map('x', 's')
        self.cluster.set_output_map('x')

        self.connect = Connect()
        self.connect.set_input_map('edge_index', 's')
        self.connect.set_output_map('edge_index')

        self.reduce = reduce
        if self.reduce is not None:
            self.reduce.set_input_map('edge_index')
            self.reduce.set_output_map('edge_index')

        self.apply_threshold = None
        self.threshold = threshold
        if self.threshold is not None:
            self.apply_threshold = ApplyThreshold(self.threshold)
            self.apply_threshold.set_input_map('edge_index')
            self.apply_threshold.set_output_map('edge_index')

        # make A sparse
        self.sparse = SparseEdgeIndex()
        self.sparse.set_input_map('edge_index')
        self.sparse.set_output_map('edge_index')

    def forward(self, x):
        x = self.select(x)
        x = self.batch_compatible(x)

        if callable(self.pool_loss):
            x = self.pool_loss(x)
        
        x = self.cluster(x)
        x = self.connect(x)

        if callable(self.reduce):
            x = self.reduce(x)

        if self.apply_threshold is not None:
            x = self.apply_threshold(x)

        x = self.sparse(x)

        return x
    

    
