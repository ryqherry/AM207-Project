import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm

class CellStateSimulation:
    def __init__(self, k1: float, k2: float, kx: float, initial_state: int):
        """
        Parameters:
        k1: switching rate from state 1 to 2
        k2: switching rate from state 2 to 1
        kx: cell proliferation rate
        initial_state: starting cell state (1 or 2)
        """
        self.k1 = k1  # State 1 -> 2 switching rate
        self.k2 = k2  # State 2 -> 1 switching rate
        self.kx = kx  # Cell proliferation rate
        
        # Initialize with single cell in given state
        self.cells_state1 = 1 if initial_state == 1 else 0
        self.cells_state2 = 1 if initial_state == 2 else 0

    def simulate(self, time: float, tau: float) -> Tuple[List[float], List[float], List[float]]:
        """
        Simulate the system using modified tau-leaping (use SSA when tau-leaping will generate negative state counts) up to given time.
        Returns final state counts.
        """
        current_time = 0
        
        while current_time < time:
            # Get possible events and their rates
            rates = []
            num_firings = []
            rates.append(self.kx * self.cells_state1)
            rates.append(self.kx * self.cells_state2)
            rates.append(self.k1 * self.cells_state1)
            rates.append(self.k2 * self.cells_state2)
            
            # Sample number of firings
            for rate in rates:
                num_firings.append(np.random.poisson(rate * tau))
            
            # Use SSA instead when tau-leaping will generate negative state counts
            if self.cells_state1 + num_firings[0] - num_firings[2] + num_firings[3] < 0 or self.cells_state2 + num_firings[1] + num_firings[2] - num_firings[3] < 0:
                # Calculate total rate
                total_rate = sum(rates)
                
                # Generate time until next event
                dt = -np.log(np.random.random()) / total_rate
                current_time += dt
                    
                # Choose event based on relative rates
                event_choice = np.random.random() * total_rate
                cumulative_rate = 0
                chosen_event = None
                
                for i, rate in enumerate(rates):
                    cumulative_rate += rate
                    if event_choice <= cumulative_rate:
                        chosen_event = i
                        break
                
                # Execute the chosen event
                if chosen_event == 0:
                    self.cells_state1 += 1
                elif chosen_event == 1:
                    self.cells_state2 += 1
                elif chosen_event == 2:
                    self.cells_state1 -= 1
                    self.cells_state2 += 1
                elif chosen_event == 3:
                    self.cells_state2 -= 1
                    self.cells_state1 += 1
            
            # Update state counts based on tau-leaping
            else:
                self.cells_state1 += num_firings[0]
                self.cells_state2 += num_firings[1]
                self.cells_state1 -= num_firings[2]
                self.cells_state2 += num_firings[2]
                self.cells_state2 -= num_firings[3]
                self.cells_state1 += num_firings[3]
            
                current_time += tau
            
        return self.cells_state1, self.cells_state2

def run_simulation(num_colonies: int, simulation_time: float, tau: float, k1: float, k2: float, kx: float, f_initial: float) -> List[float]:
    """
    Run multiple colonies and calculate fraction of cells in state 2
    """
    fractions = []
    
    for _ in range(num_colonies):
        # Choose initial state based on f_initial
        initial_state = 2 if np.random.random() < f_initial else 1
        
        # Run simulation
        simulation = CellStateSimulation(k1, k2, kx, initial_state)
        state1_counts, state2_counts = simulation.simulate(simulation_time, tau)

        # Calculate state 2 fractions
        total_cells = state1_counts + state2_counts
        fraction_state2 = state2_counts / total_cells
        fractions.append(fraction_state2)

    # Calculate squared CV across colonies    
    fractions = np.array(fractions)
    mean = np.mean(fractions)
    variance = np.var(fractions)
    cv_square = variance / (mean * mean)
            
    return cv_square

if __name__ == "__main__":
    # Parameters
    f = 0.1  # Expected fraction in state 2
    kx = 1.0  # Cell division rate
    num_colonies = 1000  # Number of colonies
    tau = 0.02  # tau for tau-leaping

    times = np.logspace(0, np.log10(50), 50)
    k2s = 1 / times  # State 2 -> 1 switching rate
    colors = ['blue', 'orange', 'green']
    
    # Run simulation
    plt.figure(figsize=(10, 6))
    for simulation_time, color in zip([8, 13, 18], colors):
        print(f"Simulating for t={simulation_time}")
        
        cv_squares = []

        for k2 in tqdm(k2s):
            k1 = k2 * f / (1-f) # Calculate k1 to achieve desired f
            cv_squares.append(run_simulation(num_colonies=num_colonies, simulation_time=simulation_time, tau=tau, k1=k1, k2=k2, kx=kx, f_initial=f))
        
        # Plot simulated results in log-log scale
        plt.loglog(times, cv_squares, label=f'Simulated t={simulation_time}', color=color)
    
    # Analytical results given by independent variable approximation
    def cv_squared_iva(t, k2, kx=1, f2=0.1):
        Z = (1 - f2) * kx / k2
        return (2 * Z * np.exp(t * kx * (Z - 2) / Z) - 2 - Z) / ((2 * np.exp(t * kx) - 1) * (Z - 2)) * (1 - f2) / (f2)
    
    # Plot analytical results
    plt.plot(times, cv_squared_iva(8, k2s), color='blue', alpha=0.3, label='Anlytical t=8')
    plt.plot(times, cv_squared_iva(13, k2s), color='orange', alpha=0.3, label='Anlytical t=13')
    plt.plot(times, cv_squared_iva(18, k2s), color='green', alpha=0.3, label='Anlytical t=18')
    plt.xlim(1, 50)
    plt.ylim(1e-4, 10)
    plt.xlabel('Time spent in State 2 (generation)')
    plt.ylabel('CV² of Fraction State 2 cells')
    plt.legend()
    plt.show()