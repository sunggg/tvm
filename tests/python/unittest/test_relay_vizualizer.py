from tvm.relay.analysis import visualize_network
from tvm import relay
import tvm.relay.testing


if __name__ == "__main__":

    mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1, dtype="float32")
    visualize_network(mod["main"], "vis-resnet-18.gv")
