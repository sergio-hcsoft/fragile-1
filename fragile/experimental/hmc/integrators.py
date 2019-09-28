import numpy as np
import torch
from torch.autograd import Variable


def to_pytorch_variable(value, grad=False):
    """casts an input to torch Variable object
    input
    -----
    value - Type: scalar, Variable object, torch.Tensor, numpy ndarray
    grad  - Type: bool . If true then we require the gradient of that object

    output
    ------
    torch.autograd.variable.Variable object
    """
    if isinstance(value, Variable):
        return value
    elif torch.is_tensor(value):
        return Variable(value, requires_grad=grad)
    elif isinstance(value, np.ndarray):
        return Variable(torch.from_numpy(value), requires_grad=grad)
    else:
        return Variable(torch.Tensor([value]), requires_grad=grad)


class Kinetic:
    """ A basic class that implements kinetic energies and computes gradients
    Methods
    -------
    gauss_ke          : Returns KE gauss
    laplace_ke        : Returns KE laplace

    Attributes
    ----------
    p    - Type       : torch.Tensor, torch.autograd.Variable,nparray
        Size       : [1, ... , N]
        Description: Vector of current momentum

    M    - Type       : torch.Tensor, torch.autograd.Variable, nparray
        Size       : \mathbb{R}^{N \times N}
        Description: The mass matrix, defaults to identity.

    """

    def __init__(self, p=None, M=None):
        if p is None and M is None:
            raise ValueError("p and M cannot be both None.")
        if M is not None:
            if isinstance(M, Variable):
                self.M = to_pytorch_variable(torch.inverse(M.data))
            else:
                self.M = to_pytorch_variable(torch.inverse(M))
        else:
            self.M = to_pytorch_variable(torch.eye(p.size()[0]))  # inverse of identity is identity

    def gauss_ke(self, p, grad=False):
        """' (p dot p) / 2 and Mass matrix M = \mathbb{I}_{dim,dim}"""
        self.p = to_pytorch_variable(p)
        P = Variable(self.p.data, requires_grad=True)
        K = 0.5 * P.t().mm(self.M).mm(P)

        if grad:
            return self.ke_gradients(P, K)
        else:
            return K

    def laplace_ke(self, p, grad=False):
        self.p = to_pytorch_variable(p)
        P = Variable(self.p.data, requires_grad=True)
        K = torch.sign(P).mm(self.M)
        if grad:
            return self.ke_gradients(P, K)
        else:
            return K

    def ke_gradients(self, P, K):
        return torch.autograd.grad([K], [P])[0]


class Integrator:
    def __init__(self, potential, min_step, max_step, max_traj, min_traj):
        self.potential = potential
        self.kinetic = None
        self.min_step = np.random.uniform(0.01, 0.07) if min_step is None else min_step
        self.max_step = np.random.uniform(0.07, 0.18) if max_step is None else max_step
        self.max_traj = np.random.uniform(18, 25) if max_traj is None else max_traj
        self.min_traj = np.random.uniform(1, 18) if min_traj is None else min_traj

    def generate_new_step_traj(self):
        """ Generates a new step adn trajectory size  """
        step_size = np.random.uniform(self.min_step, self.max_step)
        traj_size = int(np.random.uniform(self.min_traj, self.max_traj))
        return step_size, traj_size

    def leapfrog(self, p_init, q, grad_init):
        """Performs the leapfrog steps of the HMC for the specified trajectory
        length, given by num_steps
        Parameters
        ----------
            values_init
            p_init
            grad_init     - Description: contains the initial gradients of the joint w.r.t parameters.

        Outputs
        -------
            q -    Description: proposed new q
            p      -    Description: proposed new auxillary momentum
        """
        step_size, traj_size = self.generate_new_step_traj()
        values_init = q
        self.kinetic = Kinetic(p_init)
        # Start by updating the momentum a half-step and q by a full step
        p = p_init + 0.5 * step_size * grad_init
        q = values_init + step_size * self.kinetic.gauss_ke(p, grad=True)
        for i in range(traj_size - 1):
            # range equiv to [2:nsteps] as we have already performed the first step
            # update momentum
            p = p + step_size * self.potential.eval(q, grad=True)
            # update q
            q = q + step_size * self.kinetic.gauss_ke(p, grad=True)

        # Do a final update of the momentum for a half step
        p = p + 0.5 * step_size * self.potential.eval(q, grad=True)

        # return new proposal state
        return q, p
