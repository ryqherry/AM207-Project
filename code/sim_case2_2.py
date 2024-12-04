import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm

class CellStateSimulation:
    def __init__(self, k1, k2, k3, kx, initial_state):
        """
        Parameters:
        k1: switching rate from state 1 to 2
        k2: switching rate from state 2 to 1
        k3: switching rate from state 2 to 3
        kx: cell proliferation rate
        initial_state: starting cell state (1 or 2)
        """
        self.k1 = k1  # State 1 -> 2 switching rate
        self.k2 = k2  # State 2 -> 1 switching rate
        self.k3 = k3  # State 2 -> 3 switching rate
        self.kx = kx  # Cell proliferation rate
        
        # Initialize with single cell in given state
        self.cells_state1 = 1 if initial_state == 1 else 0
        self.cells_state2 = 1 if initial_state == 2 else 0
        self.cells_state3 = 0

    def simulate(self, time):
        """
        Simulate the system using SSA up to given time.
        Returns time points and state counts.
        """
        current_time = 0
        times = [0]
        state1_counts = [self.cells_state1]
        state2_counts = [self.cells_state2]
        state3_counts = [self.cells_state3]
        
        while current_time < time:
            # Get possible events and their rates
            rates = []
            rates.append(self.kx * self.cells_state1)
            rates.append(self.kx * self.cells_state2)
            rates.append(self.kx * self.cells_state3)
            rates.append(self.k1 * self.cells_state1)
            rates.append(self.k2 * self.cells_state2)
            rates.append(self.k3 * self.cells_state2)
                
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
                self.cells_state3 += 1
            elif chosen_event == 3:
                self.cells_state1 -= 1
                self.cells_state2 += 1
            elif chosen_event == 4:
                self.cells_state2 -= 1
                self.cells_state1 += 1
            elif chosen_event == 5:
                self.cells_state2 -= 1
                self.cells_state3 += 1
            
            # Record the time and state counts
            times.append(current_time)
            state1_counts.append(self.cells_state1)
            state2_counts.append(self.cells_state2)
            state3_counts.append(self.cells_state3)
            
        return times, state1_counts, state2_counts, state3_counts

def run_simulation(num_colonies, simulation_time, k1, k2, k3, kx, f_initial):
    """
    Run multiple colonies and calculate gene expressions for the case with irreversible inactive
    """
    totals = np.zeros((simulation_time+1, num_colonies))
    
    for nc in range(num_colonies):
        # Choose initial state based on f_initial
        initial_state = 1 if np.random.random() < f_initial else 2
        
        # Run simulation
        simulation = CellStateSimulation(k1, k2, k3, kx, initial_state)
        times, state1_counts, state2_counts, state3_counts = simulation.simulate(simulation_time)

        # Calculate gene expression of active cells at integer times
        timer = 0
        for i, time in enumerate(times):
            if timer > 9:
                break
            if time >= timer:
                total_active_at_time = state1_counts[i]
                totals[timer][nc] = total_active_at_time
                timer += 1
        
        # Calculate final gene expression of active cells
        total_active = state1_counts[-1]
        totals[10][nc] = total_active
            
    return totals

if __name__ == "__main__":
    # Parameters
    f = 0.1  # Expected fraction in state 1
    k1 = 0.2  # State 1 -> 2 switching rate
    kx = 1.0  # Cell division rate
    k2 = k1 * f / (1 - f)  # Calculate k2 to achieve desired f
    num_colonies = 1000  # Number of colonies
    simulation_time = 10  # Durations of colony expansion
    num_repeats = 20  # Number of repeats
    cv_squares = np.zeros((simulation_time+1, num_repeats))
    colors = ['blue', 'orange', 'green', 'purple']

    # Plot results
    plt.figure(figsize=(10, 6))
    for k3, color in zip([0, 0.02, 0.1, 0.2], colors):
        for i in tqdm(range(num_repeats), desc=f"# of repeats"):
            # Run simulation
            fraction_arr = run_simulation(num_colonies=num_colonies, simulation_time=simulation_time, k1=k1, k2=k2, k3=k3, kx=kx, f_initial=f)
            
            # Calculate CV² for this repeat
            t = 0
            for fractions in fraction_arr:
                mean = np.mean(fractions)
                variance = np.var(fractions)
                cv_square = variance / (mean * mean)
                cv_squares[t][i] = cv_square
                t += 1
        
        # Calculate CV stats across repeats for plot
        cv_mean = np.mean(cv_squares, axis=1)
        cv_std = np.std(cv_squares, axis=1)
        times = np.arange(cv_squares.shape[0])

        plt.errorbar(times, cv_mean, yerr=cv_std, fmt='o', color=color, label=fr'$k_3={k3}$', alpha=0.5, markersize=3, elinewidth=1)
        plt.xlabel('Time (generations)')
        plt.ylabel('CV² of gene expression level across colony')
        plt.legend()

    plt.show()