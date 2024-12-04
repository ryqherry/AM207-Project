import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm

class CellStateSimulation:
    def __init__(self, n, k, m, kx, initial_states):
        """
        Parameters:
        n: number of states
        k: forward switching rate
        m: backward switching rate
        kx: cell proliferation rate
        initial_states: starting cell states
        """
        self.n = n  # Number of states
        self.k = k  # Forward rate
        self.m = m  # Backward rate
        self.kx = kx  # Cell proliferation rate
        
        # Initialize with single cell in given state
        self.cells_states = initial_states

    def simulate(self, time):
        """
        Simulate the system using SSA up to given time.
        Returns time points and state counts.
        """
        current_time = 0
        times = [0]
        state_counts = [self.cells_states.copy()]
        
        while current_time < time:
            # Get possible events and their rates
            rates = []
            for i in range(self.n):
                rates.append(self.kx * self.cells_states[i])
            
            for i in range(self.n-1):
                rates.append(self.k * self.cells_states[i])
            
            for i in range(1, self.n):
                rates.append(self.m * self.cells_states[i])
                
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
            if chosen_event < self.n:
                self.cells_states[chosen_event] += 1
            elif chosen_event < self.n * 2 - 1:
                self.cells_states[chosen_event - self.n] -= 1
                self.cells_states[chosen_event - self.n + 1] += 1
            else:
                self.cells_states[chosen_event - self.n * 2 + 1] += 1
                self.cells_states[chosen_event - self.n * 2 + 2] -= 1
            
            # Record the time and state counts
            times.append(current_time)
            state_counts.append(self.cells_states.copy())
        
        return times, state_counts

def run_simulation(num_colonies, simulation_time, n, k, m, kx):
    """
    Run multiple colonies and calculate gene expressions within n-state system 
    """
    totals = np.zeros((simulation_time+1, num_colonies))

    # Generate initial state based on given n 
    prob_initial = np.ones(n)
    for i in range(1, n):
        prob_initial[i] = (k/m)**(i)
    prob_initial = prob_initial/sum(prob_initial)
    
    for nc in range(num_colonies):
        # Choose initial state based on f_initial
        indices = np.arange(n)
        sampled_index = np.random.choice(indices, p=prob_initial)

        initial_state = np.zeros(n)
        initial_state[sampled_index] = 1
        
        # Run simulation
        simulation = CellStateSimulation(n, k, m, kx, initial_state)
        times, state_counts = simulation.simulate(simulation_time)

        # Calculate gene expressions at integer time
        timer = 0
        for i, time in enumerate(times):
            if timer > 9:
                break
            if time >= timer:
                total_cells_at_time = 0
                for j, state_count in enumerate(state_counts[i]):
                    total_cells_at_time += j * state_count
                totals[timer][nc] = total_cells_at_time
                timer += 1
        
        # Calculate final gene expressions
        total_cells = 0
        for i, state_count in enumerate(state_counts[-1]):
            total_cells += i * state_count
        totals[10][nc] = total_cells
            
    return totals

if __name__ == "__main__":
    # Parameters
    kx = 1.0  # Cell division rate
    m = 0.2  # Backward rate
    k = m * 0.1 / (1 - 0.1)  # Forward rate
    num_colonies = 1000  # Number of colonies
    simulation_time = 10  # Durations of colony expansion
    num_repeats = 20  # Number of repeats
    cv_squares = np.zeros((simulation_time+1, num_repeats))
    colors = ['blue', 'orange', 'green']

    # Plot results
    plt.figure(figsize=(10, 6))
    for n, color in zip([2, 5, 10], colors):
        for i in tqdm(range(num_repeats), desc=f"# of repeats"):
            # Run simulation
            fraction_arr = run_simulation(num_colonies=num_colonies, simulation_time=simulation_time, n=n, k=k, m=m, kx=kx)
            
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

        plt.errorbar(times, cv_mean, yerr=cv_std, fmt='o', color=color, label=f'n={n}', markersize=3, elinewidth=1)
        plt.xlabel('Time (generations)')
        plt.ylabel('CV² of gene expression level across colony')
        plt.legend()

    plt.show()