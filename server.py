import flwr as fl
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights



strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl_model = fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config=fl.server.ServerConfig(num_rounds=10) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)




