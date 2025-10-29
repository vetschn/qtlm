class Device:
    
    def __init__(self):
        self.hamiltonian = None
        self.overlap = None
        self.interaction_tensor = None
        self.positions = None
        self.distances = None
        
    def compute_d0(self):
        ...

    def compute_d0_delta_perp(self):
        ...

    def configure():
        ...