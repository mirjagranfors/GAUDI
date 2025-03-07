from deeplay.components.gnn.mpn.cla import CombineLayerActivation

class TransformSender(CombineLayerActivation):
    """Transform module for MPN."""

    def get_forward_args(self, x):
        """Get the arguments for the Transform module.
        An MPN Transform module takes the following arguments:
        - node features of sender nodes (x[A[0]])
        - edge features (edgefeat)
        A is the adjacency matrix of the graph.
        """
        x, edge_index, edge_attr = x
        return x[edge_index[0]], edge_attr