from collections import OrderedDict
from torch import load
from torch.nn.utils import prune


class MaskUtils(object):
    def __init__(self):
        super(MaskUtils, self).__init__()
        self.weight_mask = OrderedDict()

    def init_mask(self, model):
        """
        Constructs initial masks for all the layers in the network. Each mask is essentially a matrix of ones.

        No masks are constructed for the biases

        Arguments
        -------
        model: (nn.Module), the feed forward network to prune
        """
        for n, m in model.named_children():
            if hasattr(m, 'weight'):
                prune.identity(m, name='weight')
                self.weight_mask[f"{n}.weight"] = m.weight_mask.detach().clone()
                prune.remove(m, name='weight')

    def prune_network(self, model, rate):
        """
        Prunes the weights of a feed forward neural network according to l1_norm.

        For LetNet300100 and CONV2, convolutional and output layers are pruned at half the rate of
        other fully connected fully connected layers. Such layers are pruned at equal layers. This follows the protocol
        in the original paper

        Arguments
        --------
        model: (nn.Module) The neural network whose layers are to be pruned.
        rate: (float) The fraction of the weights to prune. Must be between 0 and 1
        """
        for n, m in model.named_children():
            if hasattr(m, 'weight'):
                if "output" in n or 'conv' in n:
                    prune.l1_unstructured(m, 'weight', amount=rate / 2.0)
                else:
                    prune.l1_unstructured(m, 'weight', amount=rate)

                self.weight_mask[f"{n}.weight"] = m.weight_mask.detach().clone()
                prune.remove(m, name='weight')

    def freeze(self, model, weight=False):
        """
        Zero the gradients or weights of a network according to a mask. The gradients of all pruned weights are
        zeroed before sdg update step. For completeness (and perhaps waste of compute resources), double down and
        reset the weights of pruned weights to zero after sdg update step.

        Arguments
        -------
        model: (nn.Module) The network whose weights are to be zeroed according to the mask.
        weight: (bool) Indicates whether to zero the weights or the gradients according to the masks. If True,
        the weights are zeroed. The gradients of the masked weights are set to zero otherwise.
        """
        for n, m in model.named_children():
            if hasattr(m, 'weight'):
                if weight:
                    m.weight.data.mul_(self.weight_mask[f"{n}.weight"])
                else:
                    m.weight.grad.mul_(self.weight_mask[f"{n}.weight"])

    def reset_weights(self, model, initial_weights_path):
        """
        Reset the weights of surviving (unpruned) weights to their original initial values. This follows the lottery
        ticket hypothesis protocol. The masked weights will be set to 0

        Arguments:
        model: (nn.Module) The module whose surviving weights are to be reset.
        initial_weights_path: (string) The path to the initial weights of the network
        """
        try:
            model.load_state_dict(load(initial_weights_path))
        except Exception as e:
            print(f"Error restoring state dictionary: {e}")

        for n, m in model.named_children():
            if hasattr(m, 'weight'):
                m.weight.data.mul_(self.weight_mask[f"{n}.weight"])

    @staticmethod
    def num_active(model):
        """Computes the number of weights in a module which are not pruned"""
        num_active = 0
        for m in model.children():
            if hasattr(m, 'weight'):
                num_active += (m.weight.data != 0).float().sum().item()

        return num_active

    @staticmethod
    def num_weights(model):
        """Computes the total number of weights (excluding biases in a network)"""
        num_weight = 0
        for m in model.children():
            if hasattr(m, 'weight'):
                num_weight += m.weight.numel()

        return num_weight
