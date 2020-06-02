from mask_utils import Mask
from torch import nn, tensor, save


class TestMaskUtils:

    def test_prune_network(self):
        mask_utils = Mask()
        module = nn.Sequential()

        module.add_module(name="fc1", module=nn.Linear(in_features=3, out_features=4))
        module.add_module(name="output", module=nn.Linear(in_features=4, out_features=2))

        initial_weights_fc = tensor([
            [0.091, 3, 1, -0.01],
            [0.1, 4, -4, 0.01],
            [2, 0.89, 0.2, 1]
        ])

        initial_weights_output = tensor([
            [-0.02, 0.01],
            [6.0, 1],
            [0.8, -3],
            [0.09, 0.1]
        ])

        module.fc1.weight.data = initial_weights_fc.clone()
        module.output.weight.data = initial_weights_output.clone()

        rate = 0.4
        expected_mask_fc = tensor([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 1]
        ])

        # the output layer is pruned at half the rate of the other fc layers
        expected_mask_output = tensor([
            [0, 0],
            [1, 1],
            [1, 1],
            [1, 1]
        ])

        mask_utils.prune_network(module, rate)
        fc1_mask = mask_utils.weight_mask["fc1.weight"]
        output_mask = mask_utils.weight_mask['output.weight']
        assert (expected_mask_fc != fc1_mask).sum() == 0
        assert (expected_mask_output != output_mask).sum() == 0

    def test_num_active(self):
        module = nn.Sequential()
        module.add_module(name="fc1", module=nn.Linear(in_features=3, out_features=4))
        module.add_module(name="output", module=nn.Linear(in_features=4, out_features=2))
        module.fc1.weight.data = tensor([
                                            [0.091, 0, 0, -0.01],
                                            [0.1, 4, -4, 0.0],
                                            [2, 0.0, 0.2, 0.0]
                                             ])
        module.output.weight.data = tensor([
            [-0.02, 0.01],
            [0.0, 0],
            [0, -3],
            [0.09, 0.1]
        ])

        expected_num_active = 12
        num_active = Mask.num_active(module)

        assert num_active == expected_num_active

    def test_reset_weights(self):
        module = nn.Sequential()
        module.add_module(name="fc1", module=nn.Linear(in_features=2, out_features=3))

        initial_fc1__weights = tensor([
            [0.091, 3, 1],
            [0.1, 4, -4],
        ])

        module.fc1.weight.data = initial_fc1__weights
        initial_weights_path = "initial_weights.pt"
        save(module.state_dict(), initial_weights_path)

        # update the weights.
        new_weights = tensor([
            [20, 3.5, -11],
            [2, 8, -4.5],
        ])
        module.fc1.weight.data = new_weights
        assert (module.fc1.weight.data == initial_fc1__weights).sum() == 0.0

        mask_utils = Mask()
        # prune
        mask_utils.prune_network(module, rate=0.5)

        # reset weights
        mask_utils.reset_weights(module, initial_weights_path)

        expected_reset_weights = tensor([
            [0.091, 0, 1],
            [0.0, 4, 0]
        ])
        assert (module.fc1.weight.data != expected_reset_weights).sum() == 0
