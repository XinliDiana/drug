import numpy as np

############################################################
# 1. Model Parameters
############################################################

# Phase costs
c0 = 20.0   # Discovery cost
c1 = 50.0   # Phase I
c2 = 80 # Phase II
c3 = 300.0  # Phase III
c_RA = 85 # Regulatory approval cost

# Market payoff if fully approved and marketed in-house
P = 2000.0

# Probability of success if truly effective
alpha = {0: 0.8, 1: 0.8, 2: 0.6, 3: 0.7}
# Probability of false positive if not effective
beta  = {0: 0.2, 1: 0.3, 2: 0.2, 3: 0.3}

# Probability of regulatory approval if effective/ineffective
p_approve_eff  = 0.9
p_approve_fail = 0.01

# Discount factor
rho = 0.95

# Direct termination cost
C_term = 10.0

# Phase-dependent licensing payoff: L_tau(mu) = L_base[τ] + alpha_L[τ]*mu
L_base  = {0:  0.0, 1:  20.0, 2: 100.0, 3: 300.0}  # baseline
alpha_L = {0: 100.0, 1: 120.0, 2: 200.0, 3: 300.0}  # slope wrt mu

# Phase-dependent outside option: O_tau(mu) = O_base[τ] + gamma_O[τ]*mu
O_base  = {0: 50.0, 1:  40.0, 2:  30.0, 3:  20.0}
gamma_O = {0:  0.0, 1:  20.0, 2:  50.0, 3:  80.0}

# We'll define a special final licensing payoff if fully approved
L_final = 200.0

# Grid for mu
N_GRID = 101
MU_GRID = np.linspace(0,1,N_GRID)

############################################################
# 2. Helper functions
############################################################

def p_success(mu, phase):
    """Probability of success at phase given mu."""
    return mu*alpha[phase] + (1.0 - mu)*beta[phase]

def licensing_payoff(mu, phase):
    """Phase-dependent licensing payoff: L_tau(mu)."""
    return L_base[phase] + alpha_L[phase]*mu

def outside_option(mu, phase):
    """Phase-dependent outside option: O_tau(mu)."""
    return O_base[phase] + gamma_O[phase]*mu

def stop_value(mu, phase):
    """Stop payoff = max(Licensing, OutsideOption) - C_term."""
    return max( licensing_payoff(mu, phase),
                outside_option(mu, phase) ) - C_term

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
    p_app = mu*p_approve_eff + (1.0 - mu)*p_approve_fail
    return -c_RA + rho * p_app * marketing_payoff(mu)

############################################################
# 3. DP Arrays
############################################################
# We'll define 5 phases: 0..3 for the main dev, 4 for marketing
V = np.zeros((5, N_GRID))   # store value
policy = np.zeros((4, N_GRID))  # store policy for phases 0..3 (0=stop,1=cont)

# costs array for convenience
c = [c0, c1, c2, c3, 0.0]

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
    # continue => pay c3, then success => regulatory_approval_value, fail => 0
    ps = p_success(mu_val, 3)
    vc = -c3 + rho*( ps*regulatory_approval_value(mu_val) + (1-ps)*0.0 )
    V[3, i] = max(vs, vc)
    policy[3, i] = 1 if (vc > vs) else 0

# Phase 2
for i in range(N_GRID):
    mu_val = MU_GRID[i]
    vs = stop_value(mu_val, 2)
    ps = p_success(mu_val, 2)
    if ps>1e-12:
        mu_succ = (mu_val*alpha[2])/(mu_val*alpha[2] + (1.0 - mu_val)*beta[2])
    else:
        mu_succ=0.0
    # success => V[3, i_succ], fail => 0
    i_succ = int(round(mu_succ*(N_GRID-1)))
    EV_next = ps*V[3, i_succ] + (1-ps)*0.0
    vc = -c2 + rho*EV_next
    V[2, i] = max(vs, vc)
    policy[2, i] = 1 if (vc > vs) else 0

# Phase 1
for i in range(N_GRID):
    mu_val = MU_GRID[i]
    vs = stop_value(mu_val, 1)
    ps = p_success(mu_val, 1)
    if ps>1e-12:
        mu_succ = (mu_val*alpha[1])/(mu_val*alpha[1] + (1.0 - mu_val)*beta[1])
    else:
        mu_succ=0.0
    i_succ = int(round(mu_succ*(N_GRID-1)))
    EV_next = ps*V[2, i_succ] + (1-ps)*0.0
    vc = -c1 + rho*EV_next
    V[1, i] = max(vs, vc)
    policy[1, i] = 1 if (vc > vs) else 0

# Phase 0
for i in range(N_GRID):
    mu_val = MU_GRID[i]
    vs = stop_value(mu_val, 0)
    ps = p_success(mu_val, 0)
    if ps>1e-12:
        mu_succ = (mu_val*alpha[0])/(mu_val*alpha[0] + (1.0 - mu_val)*beta[0])
    else:
        mu_succ=0.0
    i_succ = int(round(mu_succ*(N_GRID-1)))
    EV_next = ps*V[1, i_succ] + (1-ps)*0.0
    vc = -c0 + rho*EV_next
    V[0, i] = max(vs, vc)
    policy[0, i] = 1 if (vc > vs) else 0

############################################################
# 6. Optional: Approximate Threshold
############################################################
def find_threshold(phase):
    pol = policy[phase,:]
    if np.all(pol==1):
        return 0.0
    if np.all(pol==0):
        return 1.0
    for i in range(N_GRID-1):
        if pol[i]==0 and pol[i+1]==1:
            return 0.5*(MU_GRID[i]+MU_GRID[i+1])
    return 1.0

thresholds = [find_threshold(ph) for ph in [0,1,2,3]]

############################################################
# 7. Visualization
############################################################
if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    axes = axes.flatten()
    for ph in [0,1,2,3]:
        ax = axes[ph]
        pol = policy[ph,:]
        ax.plot(MU_GRID, pol, drawstyle='steps-post', label=f'Policy(phase={ph})')
        th = thresholds[ph]
        ax.axvline(x=th, color='red', linestyle='--', label=f'Threshold ~ {th:.2f}')
        ax.set_ylim(-0.1,1.1)
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel("Policy (0=Stop, 1=Continue)")
        ax.legend()
    plt.suptitle("Stop/Continue Policy with Phase-Dependent Licensing & Outside Option")
    plt.tight_layout()
    plt.show()

############################################################
# 8. Explanation
############################################################
"""
We define:
- licensing_payoff(mu, phase) = L_base[phase] + alpha_L[phase]*mu
- outside_option(mu, phase)   = O_base[phase] + gamma_O[phase]*mu
- stop_value(mu, phase)       = max(licensing_payoff, outside_option) - C_term

At each phase, the DP compares V_stop vs. V_continue. 
We store the resulting value in V[phase,i], and store the policy in policy[phase,i].

Then we approximate threshold by scanning the policy array for the index 
where it flips from 0->1. We also provide a quick plot of the policy 
versus mu, plus a red line for the threshold.
"""
