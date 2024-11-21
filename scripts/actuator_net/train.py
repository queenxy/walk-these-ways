from utils import train_actuator_network_and_plot_predictions
from glob import glob

data_path = "./scripts/actuator_net/aliengo_gazebo_torques.npz"

# Evaluates the existing actuator network by default
# load_pretrained_model = True
# actuator_network_path = "../../resources/actuator_nets/unitree_go1.pt"

# Uncomment these lines to train a new actuator network
load_pretrained_model = False
actuator_network_path = "aliengo_actuator.pt"

train_actuator_network_and_plot_predictions(data_path, actuator_network_path=actuator_network_path, load_pretrained_model=load_pretrained_model)
