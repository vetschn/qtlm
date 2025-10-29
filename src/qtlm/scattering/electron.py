from qtlm.scattering.device import Device

device = Device()

class ElectronSolver:

    def __init__(self, config):
        self.config = config
        self.system_matrix = None


    def _assemble_system_matrix(self):
        # Assemble the Hamiltonian matrix based on the configuration
        pass

    def _compute_obc(self):
        # Compute the open boundary conditions
        pass

    def solve(self):
        # Main solver routine
        self._assemble_system_matrix()
        self._compute_obc()
        
        ...

