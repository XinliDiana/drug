import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

############################################################
# 1. Model Parameters
############################################################

# Phase costs - now split into fixed and variable components
c_fixed = {0: 20.0, 1: 30.0, 2: 40.0, 3: 50.0}   # Fixed costs per phase
c_var = {0: 0.0, 1: 0.1, 2: 0.2, 3: 1.0}        # Variable cost per patient

# Regulatory approval cost
c_RA = 85

# Market payoff if fully approved and marketed in-house
P = 3000.0

# Parameters for patient-dependent success probabilities
# Base probabilities
p_base_eff = {0: 0.7, 1: 0.6, 2: 0.5, 3: 0.4}   # Base prob if drug is effective
p_base_fail = {0: 0.3, 1: 0.25, 2: 0.2, 3: 0.15}  # Base prob if drug is ineffective

# Parameters for how probabilities change with patient numbers
alpha_n = {0: 0.2, 1: 0.3, 2: 0.2, 3: 0.4}      # Max improvement for effective drugs
beta_n = {0: 0.1, 1: 0.15, 2: 0.1, 3: 0.1}      # Max decrease for ineffective drugs
lambda_n = {0: 0.01, 1: 0.005, 2: 0.002, 3: 0.001}  # Rate of change for effective
delta_n = {0: 0.01, 1: 0.005, 2: 0.002, 3: 0.001}   # Rate of change for ineffective

# Probability of regulatory approval if effective/ineffective
p_approve_eff = 0.9
p_approve_fail = 0.01

# Discount factor
rho = 0.95

# Direct termination cost
C_term = 10.0

# Phase-dependent licensing payoff: L_tau(mu) = L_base[τ] + alpha_L[τ]*mu
L_base = {0: 0.0, 1: 10.0, 2: 50.0, 3: 200.0}  # Reduced from 0/20/100/300
alpha_L = {0: 40.0, 1: 80.0, 2: 150.0, 3: 300.0}  # Reduced from 100/120/200/300

# Phase-dependent outside option: O_tau(mu) = O_base[τ] + gamma_O[τ]*mu
O_base = {0: 10.0, 1: 15.0, 2: 20.0, 3: 25.0}
gamma_O = {0: 0.0, 1: 10.0, 2: 20.0, 3: 30.0}

# We'll define a special final licensing payoff if fully approved
L_final = 200.0

# Grid for mu
N_GRID = 101
MU_GRID = np.linspace(0, 1, N_GRID)

# Patient number grid - from 0 to 1000 patients
N_PATIENTS = 21
PATIENT_GRID = np.linspace(0, 1000, N_PATIENTS)

############################################################
# 2. Helper functions
############################################################

def p_effective(mu, phase, n_patients):
    """Probability of success if drug is effective, with n_patients."""
    return p_base_eff[phase] + alpha_n[phase] * (1 - np.exp(-lambda_n[phase] * n_patients))

def p_ineffective(mu, phase, n_patients):
    """Probability of false positive if drug is ineffective, with n_patients."""
    return p_base_fail[phase] - beta_n[phase] * (1 - np.exp(-delta_n[phase] * n_patients))

def p_success(mu, phase, n_patients):
    """Probability of success at phase given mu and n_patients."""
    p_eff = p_effective(mu, phase, n_patients)
    p_ineff = p_ineffective(mu, phase, n_patients)
    return mu * p_eff + (1.0 - mu) * p_ineff

def update_belief(mu, phase, n_patients):
    """Update belief after a successful outcome."""
    p_eff = p_effective(mu, phase, n_patients)
    p_ineff = p_ineffective(mu, phase, n_patients)
    if (mu * p_eff + (1.0 - mu) * p_ineff) > 1e-12:
        return (mu * p_eff) / (mu * p_eff + (1.0 - mu) * p_ineff)
    else:
        return 1.0

def phase_cost(phase, n_patients):
    """Cost function dependent on patient numbers."""
    return c_fixed[phase] + c_var[phase] * n_patients

def licensing_payoff(mu, phase):
    """Phase-dependent licensing payoff: L_tau(mu)."""
    return L_base[phase] + alpha_L[phase] * mu

def outside_option(mu, phase):
    """Phase-dependent outside option: O_tau(mu)."""
    return O_base[phase] + gamma_O[phase] * mu

def stop_value(mu, phase):
    """Stop payoff = max(Licensing, OutsideOption) - C_term."""
    return max(licensing_payoff(mu, phase),
               outside_option(mu, phase)) - C_term

def marketing_payoff(mu):
    """
    Once at final marketing stage, pick best among:
    P, L_final, or 0. 
    """
    return max(P, L_final, 0.0)

def regulatory_approval_value(mu):
    """
    If we attempt approval at mu, cost c_RA, 
    approval prob = mu*p_approve_eff + (1-mu)*p_approve_fail,
    then marketing payoff.
    """
    p_app = mu * p_approve_eff + (1.0 - mu) * p_approve_fail
    return -c_RA + rho * p_app * marketing_payoff(mu)

############################################################
# 3. DP Arrays
############################################################
# We'll define 5 phases: 0..3 for the main dev, 4 for marketing
V = np.zeros((5, N_GRID))  # store value
policy = np.zeros((4, N_GRID))  # store policy for phases 0..3 (0=stop,1=cont)
opt_patients = np.zeros((4, N_GRID))  # store optimal patient numbers

############################################################
# 4. Fill V for marketing (phase 4)
############################################################
for i in range(N_GRID):
    mu_val = MU_GRID[i]
    V[4, i] = marketing_payoff(mu_val)

############################################################
# 5. Backward Induction
############################################################

# Phase 3
for i in range(N_GRID):
    mu_val = MU_GRID[i]
    vs = stop_value(mu_val, 3)
    
    # Find optimal patient number for Phase 3
    best_vc = -float('inf')
    best_n = 0
    
    for n_idx, n_patients in enumerate(PATIENT_GRID):
        # continue => pay c3(n), then success => regulatory_approval_value, fail => 0
        ps = p_success(mu_val, 3, n_patients)
        vc_n = -phase_cost(3, n_patients) + rho * (ps * regulatory_approval_value(mu_val) + (1 - ps) * 0.0)
        
        if vc_n > best_vc:
            best_vc = vc_n
            best_n = n_patients
    
    V[3, i] = max(vs, best_vc)
    policy[3, i] = 1 if (best_vc > vs) else 0
    opt_patients[3, i] = best_n if (best_vc > vs) else 0

# Phase 2
for i in range(N_GRID):
    mu_val = MU_GRID[i]
    vs = stop_value(mu_val, 2)
    
    # Find optimal patient number for Phase 2
    best_vc = -float('inf')
    best_n = 0
    
    for n_idx, n_patients in enumerate(PATIENT_GRID):
        ps = p_success(mu_val, 2, n_patients)
        mu_succ = update_belief(mu_val, 2, n_patients)
        
        # Find index in mu grid for updated belief
        i_succ = int(round(mu_succ * (N_GRID - 1)))
        i_succ = max(0, min(i_succ, N_GRID - 1))  # Ensure valid index
        
        # success => V[3, i_succ], fail => 0
        EV_next = ps * V[3, i_succ] + (1 - ps) * 0.0
        vc_n = -phase_cost(2, n_patients) + rho * EV_next
        
        if vc_n > best_vc:
            best_vc = vc_n
            best_n = n_patients
    
    V[2, i] = max(vs, best_vc)
    policy[2, i] = 1 if (best_vc > vs) else 0
    opt_patients[2, i] = best_n if (best_vc > vs) else 0

# Phase 1
for i in range(N_GRID):
    mu_val = MU_GRID[i]
    vs = stop_value(mu_val, 1)
    
    # Find optimal patient number for Phase 1
    best_vc = -float('inf')
    best_n = 0
    
    for n_idx, n_patients in enumerate(PATIENT_GRID):
        ps = p_success(mu_val, 1, n_patients)
        mu_succ = update_belief(mu_val, 1, n_patients)
        
        # Find index in mu grid for updated belief
        i_succ = int(round(mu_succ * (N_GRID - 1)))
        i_succ = max(0, min(i_succ, N_GRID - 1))  # Ensure valid index
        
        # success => V[2, i_succ], fail => 0
        EV_next = ps * V[2, i_succ] + (1 - ps) * 0.0
        vc_n = -phase_cost(1, n_patients) + rho * EV_next
        
        if vc_n > best_vc:
            best_vc = vc_n
            best_n = n_patients
    
    V[1, i] = max(vs, best_vc)
    policy[1, i] = 1 if (best_vc > vs) else 0
    opt_patients[1, i] = best_n if (best_vc > vs) else 0

# Phase 0
for i in range(N_GRID):
    mu_val = MU_GRID[i]
    vs = stop_value(mu_val, 0)
    
    # Find optimal patient number for Phase 0
    best_vc = -float('inf')
    best_n = 0
    
    for n_idx, n_patients in enumerate(PATIENT_GRID):
        ps = p_success(mu_val, 0, n_patients)
        mu_succ = update_belief(mu_val, 0, n_patients)
        
        # Find index in mu grid for updated belief
        i_succ = int(round(mu_succ * (N_GRID - 1)))
        i_succ = max(0, min(i_succ, N_GRID - 1))  # Ensure valid index
        
        # success => V[1, i_succ], fail => 0
        EV_next = ps * V[1, i_succ] + (1 - ps) * 0.0
        vc_n = -phase_cost(0, n_patients) + rho * EV_next
        
        if vc_n > best_vc:
            best_vc = vc_n
            best_n = n_patients
    
    V[0, i] = max(vs, best_vc)
    policy[0, i] = 1 if (best_vc > vs) else 0
    opt_patients[0, i] = best_n if (best_vc > vs) else 0

# Debug output for initial belief
initial_mu = 0.6
idx = int(round(initial_mu * (N_GRID - 1)))
vs = stop_value(initial_mu, 0)
license_val = licensing_payoff(initial_mu, 0)
outside_val = outside_option(initial_mu, 0)
print(f"At μ={initial_mu}, Phase 0:")
print(f"  Stop value: {vs}")
print(f"  └─ Licensing payoff: {license_val}")
print(f"  └─ Outside option: {outside_val}")
print(f"  └─ Termination cost: {C_term}")

n_opt = opt_patients[0, idx]
ps = p_success(initial_mu, 0, n_opt)
cost = phase_cost(0, n_opt)
mu_next = update_belief(initial_mu, 0, n_opt)
i_next = int(round(mu_next * (N_GRID - 1)))
vc = -cost + rho * ps * V[1, i_next]
print(f"  Continue value: {vc} with {n_opt} patients")
print(f"    └─ Cost: {cost}")
print(f"    └─ Success probability: {ps}")
print(f"    └─ Updated belief if success: {mu_next}")
print(f"    └─ Value at next phase: {V[1, i_next]}")
print(f"  Policy decision: {'Continue' if policy[0,idx] == 1 else 'Stop'}")

############################################################
# 6. Optional: Approximate Threshold
############################################################
def find_threshold(phase):
    pol = policy[phase, :]
    if np.all(pol == 1):
        return 0.0
    if np.all(pol == 0):
        return 1.0
    for i in range(N_GRID - 1):
        if pol[i] == 0 and pol[i + 1] == 1:
            return 0.5 * (MU_GRID[i] + MU_GRID[i + 1])
    return 1.0

thresholds = [find_threshold(ph) for ph in [0, 1, 2, 3]]

############################################################
# 7. Visualization
############################################################
if __name__ == "__main__":
    # Create a figure with two rows of plots
    fig = plt.figure(figsize=(15, 10))
    
    # First row: Stop/Continue policy as a function of μ (as before)
    for ph in range(4):
        ax = fig.add_subplot(2, 4, ph + 1)
        pol = policy[ph, :]
        ax.plot(MU_GRID, pol, drawstyle='steps-post', label=f'Policy(phase={ph})')
        th = thresholds[ph]
        ax.axvline(x=th, color='red', linestyle='--', label=f'Threshold ~ {th:.2f}')
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel("Policy (0=Stop, 1=Continue)")
        ax.set_title(f"Phase {ph} Decision")
        ax.legend()
    
    # Second row: Optimal patient numbers as a function of μ
    for ph in range(4):
        ax = fig.add_subplot(2, 4, ph + 5)
        # Only show patient numbers where policy is to continue
        masked_patients = np.copy(opt_patients[ph, :])
        masked_patients[policy[ph, :] == 0] = np.nan  # Set to NaN where policy is to stop
        
        ax.plot(MU_GRID, masked_patients, '-', label=f'Phase {ph}')
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel("Optimal Patient Number")
        ax.set_title(f"Phase {ph} Optimal Patients")
        ax.grid(True)
        
        # Add threshold line
        th = thresholds[ph]
        ax.axvline(x=th, color='red', linestyle='--')
        ax.set_ylim(0, PATIENT_GRID.max() * 1.1)  # Ensure proper y-axis limits
    
    plt.suptitle("Drug Development Policy with Patient Number Optimization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.show()
    
    # Create a 3D visualization of value function across μ and patient numbers for Phase 2
    phase_to_visualize = 2  # Change this to visualize different phases
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of μ and patient numbers
    mu_mesh, n_mesh = np.meshgrid(MU_GRID, PATIENT_GRID)
    v_mesh = np.zeros_like(mu_mesh)
    
    # Calculate value function at each point
    for i in range(len(MU_GRID)):
        for j in range(len(PATIENT_GRID)):
            mu_val = MU_GRID[i]
            n_val = PATIENT_GRID[j]
            
            # Calculate stop value
            vs = stop_value(mu_val, phase_to_visualize)
            
            # Calculate continue value with this patient number
            ps = p_success(mu_val, phase_to_visualize, n_val)
            mu_succ = update_belief(mu_val, phase_to_visualize, n_val)
            i_succ = int(round(mu_succ * (N_GRID - 1)))
            i_succ = max(0, min(i_succ, N_GRID - 1))
            
            # For phase 2, the next phase is 3
            next_phase = phase_to_visualize + 1
            EV_next = ps * V[next_phase, i_succ] + (1 - ps) * 0.0
            vc = -phase_cost(phase_to_visualize, n_val) + rho * EV_next
            
            # Store the maximum of stop and continue values
            v_mesh[j, i] = max(vs, vc)
    
    # Plot the surface
    surf = ax.plot_surface(mu_mesh, n_mesh, v_mesh, cmap=cm.viridis,
                          linewidth=0, antialiased=False, alpha=0.8)
    ax.set_xlabel('Belief (μ)')
    ax.set_ylabel('Number of Patients')
    ax.set_zlabel('Value')
    ax.set_title(f'Phase {phase_to_visualize} Value Function')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

############################################################
# 8. Analysis Tools: Expected Development Path
############################################################
def simulate_development_path(initial_belief, num_simulations=1000):
    """
    Simulate the drug development process starting with an initial belief.
    Returns statistics about successful paths.
    """
    completed_phases = np.zeros(num_simulations, dtype=int)
    total_patients = np.zeros(num_simulations, dtype=float)
    total_costs = np.zeros(num_simulations, dtype=float)
    final_beliefs = np.zeros(num_simulations, dtype=float)
    
    for sim in range(num_simulations):
        mu = initial_belief
        phase = 0
        patients_used = 0
        cost_incurred = 0
        
        while phase < 4:
            # Find the current policy and patient numbers
            mu_idx = int(round(mu * (N_GRID - 1)))
            mu_idx = max(0, min(mu_idx, N_GRID - 1))
            
            if policy[phase, mu_idx] == 0:  # Stop
                break
            
            # Get optimal patient number
            n_patients = opt_patients[phase, mu_idx]
            patients_used += n_patients
            
            # Incur cost
            phase_cost_val = phase_cost(phase, n_patients)
            cost_incurred += phase_cost_val
            
            # Determine if trial is successful (simulate true state of the drug)
            is_effective = np.random.random() < mu  # True with probability mu
            if is_effective:
                p_success_val = p_effective(mu, phase, n_patients)
            else:
                p_success_val = p_ineffective(mu, phase, n_patients)
            
            trial_success = np.random.random() < p_success_val
            if not trial_success:
                break  # Failed trial
            
            # Update belief and move to next phase
            mu = update_belief(mu, phase, n_patients)
            phase += 1
            
            # If we've reached regulatory approval, add that cost too
            if phase == 4:
                # Regulatory approval phase
                approval_success = np.random.random() < (mu * p_approve_eff + (1 - mu) * p_approve_fail)
                cost_incurred += c_RA
                if not approval_success:
                    break
        
        # Record results
        completed_phases[sim] = phase
        total_patients[sim] = patients_used
        total_costs[sim] = cost_incurred
        final_beliefs[sim] = mu
    
    # Compute statistics
    success_rate = np.mean(completed_phases == 4)
    avg_patients = np.mean(total_patients)
    avg_cost = np.mean(total_costs)
    phases_reached = {i: np.mean(completed_phases >= i) for i in range(1, 5)}
    
    return {
        'success_rate': success_rate,
        'avg_patients': avg_patients,
        'avg_cost': avg_cost,
        'phases_reached': phases_reached,
        'completed_phases': completed_phases,
        'total_patients': total_patients,
        'total_costs': total_costs,
        'final_beliefs': final_beliefs
    }

# Example usage of the simulation
if __name__ == "__main__":
    initial_belief = 0.6  # Starting belief about drug efficacy
    results = simulate_development_path(initial_belief)
    
    print(f"Starting with initial belief μ = {initial_belief}:")
    print(f"Overall success rate: {results['success_rate']:.2%}")
    print(f"Average patients enrolled: {results['avg_patients']:.1f}")
    print(f"Average development cost: ${results['avg_cost']:.1f}M")
    print("\nPhase progression rates:")
    for phase, rate in results['phases_reached'].items():
        print(f"  Reached Phase {phase}: {rate:.2%}")
    
    # Plot distribution of patients used
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(results['total_patients'], bins=20, alpha=0.7)
    plt.xlabel('Total Patients Enrolled')
    plt.ylabel('Frequency')
    plt.title('Distribution of Patient Numbers')
    
    plt.subplot(1, 2, 2)
    plt.hist(results['total_costs'], bins=20, alpha=0.7)
    plt.xlabel('Total Development Cost')
    plt.ylabel('Frequency')
    plt.title('Distribution of Development Costs')
    plt.tight_layout()
    plt.show()