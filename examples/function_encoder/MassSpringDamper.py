import torch
import matplotlib.pyplot as plt

from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.dynamics.integrators import Integrator
from neuromancer.modules import blocks
from neuromancer.system import Node
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from tqdm import trange

from FunctionEncoder import FunctionEncoder

def springmassdamper(state, zeta, omega_n):
    """The spring mass damper mapping from x to \dot x for given values zeta and omega_n. """
    """ zeta: damping ratio, omega_n: natural frequency """
    x1 = state[..., 0]  # position
    x2 = state[..., 1]  # velocity
    dx1 = -omega_n ** 2 * x1 - 2 * zeta * omega_n * x2
    dx2 = x2
    return torch.stack([dx1, dx2], dim=-1)

def rk4_delta_only(model, state, dt):
    """
    Integrate a model using the 4th order Runge-Kutta method.
    :param model: a callable mapping from x to \dot x
    :param state: The initial state
    :param dt:  The time horizon
    :return: Delta X, the change in state
    """
    k1 = model(state)
    k2 = model(state + dt / 2 * k1)
    k3 = model(state + dt / 2 * k2)
    k4 = model(state + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dt = 0.05
print("Device: ", device)

def create_dataset(device='cpu', generate_trajectory = False):

    # hyper parameters
    n_functions = 10


    # input space for sampling
    input_low = torch.tensor([-4., -4.], device=device)
    input_high = torch.tensor([4., 4.], device=device)

    # spring mass damper equation is only valid for damping ratio between 0 and 1
    zeta_range = torch.tensor([0, 1], device=device)
    omega_n_range = torch.tensor([0.1, 3.], device=device)

    # samples dynamical systems
    zetas = torch.rand(n_functions, 1, device=device) * (zeta_range[1] - zeta_range[0]) + zeta_range[0]
    omega_ns = torch.rand(n_functions, 1, device=device) * (omega_n_range[1] - omega_n_range[0]) + omega_n_range[0]

    # we can generate a series of trajectories with different values of zeta and omega_n.
    if generate_trajectory:
        time_horizon = 1000
        
        # shape of the initial state is (n_functions, 1, 2)
        # the singleton dimension is the time dimension (a single state at time zero) and makes it easier to concatenate the states together
        initial_states = torch.rand(n_functions, 1, 2, device=device) * (input_high - input_low) + input_low
        example_states = [initial_states]       
        for i in range(time_horizon):
            current_state = example_states[-1]
            delta_X = rk4_delta_only(lambda x: springmassdamper(x, zetas, omega_ns), current_state, dt)
            new_state = current_state + delta_X
            example_states.append(new_state)
        example_states = torch.cat(example_states, dim=1) # shape (n_functions, time_horizon + 1, 2)

        # Generate a trajectory of query data for each zeta and omega_n
        initial_states = torch.rand(n_functions, 1, 2, device=device) * (input_high - input_low) + input_low
        query_states = [initial_states]
        for i in range(time_horizon):
            current_state = query_states[-1]
            delta_X = rk4_delta_only(lambda x: springmassdamper(x, zetas, omega_ns), current_state, dt)
            new_state = current_state + delta_X
            query_states.append(new_state)
        query_states = torch.cat(query_states, dim=1)

        # now create the dataset for the function encoder. The input is a state x, the output is the delta x
        example_xs = example_states[:, :-1, :]
        example_delta_xs = example_states[:, 1:, :] - example_states[:, :-1, :]
        query_xs = query_states[:, :-1, :]
        delta_xs = query_states[:, 1:, :] - query_states[:, :-1, :]

        dataset = {"example_xs": example_xs.to(device),
                    "example_ys": example_delta_xs.to(device),
                    "query_xs": query_xs.to(device),
                    "query_ys": delta_xs.to(device),
                    "zetas": zetas.to(device),
                    "omega_ns": omega_ns.to(device)
                }
    else:
        # sample uniformly in the input space, then take one step
        n_datapoints = 1000
        # Generate a trajectory of example data for each zeta and omega_n
        example_xs = torch.rand(n_functions, n_datapoints, 2, device=device) * (input_high - input_low) + input_low
        query_xs = torch.rand(n_functions, n_datapoints, 2, device=device) * (input_high - input_low) + input_low
        
        # compute next state
        example_delta_xs = rk4_delta_only(lambda x: springmassdamper(x, zetas, omega_ns), example_xs, dt)
        query_delta_xs = rk4_delta_only(lambda x: springmassdamper(x, zetas, omega_ns), query_xs, dt)

        dataset = {"example_xs": example_xs.to(device), 
                    "example_ys": example_delta_xs.to(device),
                    "query_xs": query_xs.to(device),
                    "query_ys": query_delta_xs.to(device),
                    "zetas": zetas.to(device),
                    "omega_ns": omega_ns.to(device)
                }
    return dataset



# Create dataset
train_dataset = create_dataset(device, generate_trajectory=True)
test_dataset = create_dataset(device, generate_trajectory=True)

# Wrap data into Neuromancer DictDatasets
train_data = DictDataset(train_dataset, name='train')
dev_data = DictDataset(test_dataset, name='dev')

# Create torch dataloaders with DictDatasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10,
                                           collate_fn=train_data.collate_fn,
                                           shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=10,
                                         collate_fn=dev_data.collate_fn,
                                         shuffle=False)

# Initialize basis functions networks
n_basis = 11
basis_functions = [
    blocks.MLP(2, 2,
                 linear_map=torch.nn.Linear, # for the first layer (map to a hidden rep without distorting the signal with nonlinearity) and last layer (map to the output and allow full continuous ranges, whereas nonlinear activation forces the output to be in a limited range)
                 nonlin=torch.nn.ReLU, # for the hidden layers to learn complex mappings
                 hsizes=[100]).to(device)  # hidden layer size
    for i in range(n_basis)]

# use an integrator to make them neural ODEs
class RK4_delta_only(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        h = self.h
        k1 = self.block(x, *args)                    # k1 = f(x_i, t_i)
        k2 = self.block(x + h*k1/2.0, *args)         # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        k3 = self.block(x + h*k2/2.0, *args)         # k3 = f(x_i + 0.5*h*k2, t_i + 0.5*h)
        k4 = self.block(x + h*k3, *args)             # k4 = f(y_i + h*k3, t_i + h)
        return h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)
basis_functions = [RK4_delta_only(f, h=dt) for f in basis_functions]

# the function encoder class provides convenient methods to calibrate estimates using least squares
# specifically, it uses example data to compute the coefficients of the basis functions,
# then uses a linear combination of the basis functions to estimate the function
function_encoder = FunctionEncoder(basis_functions)

# Symbolic wrapper of the neural nets
function_encoder = Node(function_encoder, ['example_xs', 'example_ys', 'query_xs'], ['query_y_hats', 'gram'], name='function_encoder')

# Define symbolic variables in Neuromancer
# these are the variables used in the loss functions
query_y_hats = variable('query_y_hats')
query_ys = variable('query_ys')
gram = variable('gram')

# Define the losses
# The first ensures the function estimates align with the true function
loss_data = (query_y_hats == query_ys)^2
loss_data.name = "loss_data"

# this prevents the magnitude of the basis functions from growing
loss_gram = (torch.diagonal(gram, dim1=1, dim2=2) == torch.ones_like(torch.diagonal(gram, dim1=1, dim2=2)))^2
loss_gram.name = "loss_gram"

# add the two losses together
loss = PenaltyLoss(objectives=[loss_data, loss_gram], constraints=[])


# Construct the optimization problems
problem = Problem(nodes=[function_encoder], loss=loss)


# Create trainer
num_epochs = 200
trainer = Trainer(
    problem.to(device),
    train_data=train_loader,
    dev_data=dev_loader,
    optimizer=torch.optim.Adam(problem.parameters(), lr=1e-3),
    epoch_verbose=10,
    epochs=num_epochs,
    warmup=num_epochs,
    device=device,
)


# Train function encoder
best_model = trainer.train()
#
# # get best model
problem.load_state_dict(best_model)
trained_model = problem.nodes[0]
trained_model = trained_model.callable

# estimate outputs and plot.
with torch.no_grad():
    example_xs, example_ys, query_xs, query_ys, zetas, omega_ns = test_dataset["example_xs"], test_dataset["example_ys"], test_dataset["query_xs"], test_dataset["query_ys"], test_dataset["mu"]
    fig, axs = plt.subplots(3,3, figsize=(15,10))
    coefficients, gram = trained_model.compute_representation(example_xs, example_ys)

    # rollout a trajectory starting from the first query xs for each environment
    estimated_states = [query_xs[:, 0:1, :]]
    for i in trange(1, query_xs.shape[1], desc="Rolling out trajectory"):
        current_state = estimated_states[-1]
        delta_X = trained_model.predict(current_state, coefficients)
        new_state = current_state + delta_X
        estimated_states.append(new_state)
    estimated_states = torch.cat(estimated_states, dim=1)


    for i in range(9):
        ax = axs[i//3, i%3]
        # plot the ground truth van der pol
        ax.plot(query_xs[i, :, 0].cpu().numpy(), query_xs[i, :, 1].cpu().numpy(), label="Ground truth", color="black", ls="--")

        # plot the estimated van der pol
        ax.plot(estimated_states[i, :, 0].cpu().numpy(), estimated_states[i, :, 1].cpu().numpy(), label="Estimated", color="red")

        # add labels
        if i == 0:
            ax.legend()
        if i % 3 == 0:
            ax.set_ylabel("x2")
        if i // 3 == 2:
            ax.set_xlabel("x1")

        # add parameters as title
        zeta = zetas[i].cpu().item()
        omega_n = omega_ns[i].cpu().item()
        ax.set_title(f"$\zeta = {zeta:.1f}, \omega_n = {omega_n:.1f}$")
    

        # set lims
        ax.set_xlim([-4, 4])
        ax.set_ylim([-5, 5])


    plt.savefig("massSpringDamper.png")