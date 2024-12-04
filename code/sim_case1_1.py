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

    def simulate(self, time: float) -> Tuple[List[float], List[float], List[float]]:
        """
        Simulate the system using SSA up to given time.
        Returns time points and state counts.
        """
        current_time = 0
        times = [0]
        state1_counts = [self.cells_state1]
        state2_counts = [self.cells_state2]
        
        while current_time < time:
            # Get possible events and their rates
            rates = []
            rates.append(self.kx * self.cells_state1)
            rates.append(self.kx * self.cells_state2)
            rates.append(self.k1 * self.cells_state1)
            rates.append(self.k2 * self.cells_state2)
                
            # Calculate total rate
            total_rate = sum(rates)
            
            # Generate time until next event
            dt = -np.log(np.random.random()) / total_rate
            current_time += dt
            
            if current_time > time:
                break
                
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
            
            # Record the time and state counts
            times.append(current_time)
            state1_counts.append(self.cells_state1)
            state2_counts.append(self.cells_state2)
            
        return times, state1_counts, state2_counts

def run_simulation(num_colonies: int, simulation_time: float, k1: float, k2: float, kx: float, f_initial: float) -> List[float]:
    """
    Run multiple colonies and calculate fraction of cells in state 2
    """
    fractions = np.zeros((simulation_time+1, num_colonies))
    
    for nc in range(num_colonies):
        # Choose initial state based on f_initial
        initial_state = 2 if np.random.random() < f_initial else 1
        
        # Run simulation
        simulation = CellStateSimulation(k1, k2, kx, initial_state)
        times, state1_counts, state2_counts = simulation.simulate(simulation_time)

        # Calculate state fractions at integer times
        timer = 0
        for i, time in enumerate(times):
            if timer > 9:
                break
            if time >= timer:
                total_cells_at_time = state1_counts[i] + state2_counts[i]
                fraction_state2_at_time = state2_counts[i] / total_cells_at_time
                fractions[timer][nc] = fraction_state2_at_time
                timer += 1
        
        # Calculate final fraction in state 2
        total_cells = state1_counts[-1] + state2_counts[-1]
        if total_cells > 0:
            fraction_state2 = state2_counts[-1] / total_cells
            fractions[10][nc] = fraction_state2
            
    return fractions

if __name__ == "__main__":
    # Parameters
    f = 0.1  # Expected fraction in state 2
    k2 = 0.2  # State 2 -> 1 switching rate
    kx = 1.0  # Cell division rate
    k1 = k2 * f / (1 - f)  # Calculate k1 to achieve desired f
    num_colonies = 1000  # Number of colonies
    simulation_time = 10  # Duration of colony expansion
    num_repeats = 20  # Number of repeats
    cv_squares = np.zeros((simulation_time+1, num_repeats))
    
    for i in tqdm(range(num_repeats), desc=f"# of repeats"):
        # Run simulation for this repeat
        fraction_arr = run_simulation(num_colonies=num_colonies, simulation_time=simulation_time, k1=k1, k2=k2, kx=kx, f_initial=f)
        
        # Calculate CV² for this repeat
        t = 0
        for fractions in fraction_arr:
            mean = np.mean(fractions)
            variance = np.var(fractions)
            cv_square = variance / (mean * mean)
            cv_squares[t][i] = cv_square
            t += 1
    
    # Caculate CV stats over repeats for plot
    cv_mean = np.mean(cv_squares, axis=1)
    cv_std = np.std(cv_squares, axis=1)
    times = np.arange(cv_squares.shape[0])

    # Analytical results given by independent variable approximation
    def cv_squared_iva(t, kx=1, f2=0.1, k2=0.2):
        Z = (1 - f2) * kx / k2
        return (2 * Z * np.exp(t * kx * (Z - 2) / Z) - 2 - Z) / ((2 * np.exp(t * kx) - 1) * (Z - 2)) * (1 - f2)/(f2)
    
    # Times for plotting analytical result
    times_ana = np.linspace(0, 10, 100)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(times, cv_mean, yerr=cv_std, fmt='o', color='blue', label='Simulations', markersize=3, elinewidth=1)
    plt.plot(times_ana, cv_squared_iva(times_ana), label='Independent Variable Approximation', color='orange')
    plt.xlabel('Time (generations)')
    plt.ylabel('CV² of Fraction State 2 cells')
    plt.legend()
    plt.show()