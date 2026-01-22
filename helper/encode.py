import numpy as np

class freq_encoder:
    """
    Frequency (rate) encoder for DSQN state:
      state = [Theta_o (1..8), S_o (0..4), Theta_g (1..8), beta (1..8)]
    Produces Poisson spikes over T timesteps for a one-hot population per field.
    Output shape: (T, 8 + 5 + 8 + 8) = (T, 29)
    """
    def __init__(self, max_freq=200.0, num_timesteps=100, dt=1e-3,
                 bg_freq=0.0, noise_std=0.0, seed=None):
        self.max_freq = float(max_freq)
        self.num_timesteps = int(num_timesteps)
        self.dt = float(dt)
        self.bg_freq = float(bg_freq)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

        # field sizes in fixed order
        # [Theta_o, S_o, Theta_g, beta]
        self.sizes = [8, 5, 8, 8]   
        self.total = sum(self.sizes)

    def _onehot_indices(self, state):
        """
        Convert state values to 0-indexed positions.
        - Theta_o: 1-8 → 0-7
        - S_o: 0-4 → 0-4  
        - Theta_g: 1-8 → 0-7
        - beta: 1-8 → 0-7
        """
        indices = []
        for val, size in zip(state, self.sizes):
            val = int(val)
            # For fields with range 1-N, subtract 1 to get 0-indexed
            # For S_o with range 0-4, keep as is
            if size == 5:  # S_o field
                idx = val
            else:  # Theta_o, Theta_g, beta (all 1-8)
                idx = val - 1
            
            # Clamp to valid range
            idx = max(0, min(idx, size - 1))
            indices.append(idx)
        
        return indices

    def _rate_vector(self, state):
        """Build per-neuron firing rates (Hz) for the 29 inputs."""
        i_o, i_s, i_g, i_b = self._onehot_indices(state)
        rates = np.full(self.total, self.bg_freq, dtype=float)

        # offsets of each field in the concatenated vector
        off_o = 0
        off_s = off_o + 8
        off_g = off_s + 5
        off_b = off_g + 8

        rates[off_o + i_o] = self.max_freq
        rates[off_s + i_s] = self.max_freq
        rates[off_g + i_g] = self.max_freq
        rates[off_b + i_b] = self.max_freq

        # optional multiplicative Gaussian noise on rates - not used
        if self.noise_std > 0:
            noise = self.rng.normal(loc=1.0, scale=self.noise_std, size=rates.shape)
            rates = np.clip(rates * noise, 0.0, None)
        return rates

    def encode_state(self, state, deterministic=False):
        """
        Return Poisson spikes as a boolean array of shape (T, 29).
        Spike prob per step for neuron j: p_j = 1 - exp(-rate_j * dt)
        """
        rates = self._rate_vector(state)  # Hz
        p = 1.0 - np.exp(-rates * self.dt)
        p = np.clip(p, 0.0, 1.0)

        if deterministic:
            # Use expected spike probability as continuous values
            # Repeat the probability pattern over time steps
            spikes_continuous = np.tile(p, (self.num_timesteps, 1))
            return spikes_continuous.astype(np.float32)
        else:
            # Original random Poisson sampling
            spikes = self.rng.uniform(size=(self.num_timesteps, self.total)) < p
            return spikes.astype(np.uint8)


def encode_one_hot(state):
    """
    Converts the 4-element state [R_o, D_o, R_T, A_T_o] into a 
    one-hot vector of length 29.
    
    Structure:
    - R_o (Obs Region): 1-8   -> 8 neurons
    - D_o (Dyn Dir):    0-4   -> 5 neurons
    - R_T (Goal Reg):   1-8   -> 8 neurons
    - A_T (Angle):      1-8   -> 8 neurons
    Total = 29
    """
    r_o, d_o, r_t, a_t = state.astype(int)
    
    # Create zero vectors
    vec_ro = np.zeros(8)
    vec_do = np.zeros(5)
    vec_rt = np.zeros(8)
    vec_at = np.zeros(8)
    
    # Set active indices (adjusting 1-based indices to 0-based)
    # R_o is 1..8 -> idx 0..7
    if 1 <= r_o <= 8: vec_ro[r_o - 1] = 1.0
    # D_o is 0..4 -> idx 0..4
    if 0 <= d_o <= 4: vec_do[d_o] = 1.0
    # R_T is 1..8 -> idx 0..7
    if 1 <= r_t <= 8: vec_rt[r_t - 1] = 1.0
    # A_T is 1..8 -> idx 0..7
    if 1 <= a_t <= 8: vec_at[a_t - 1] = 1.0
    
    return np.concatenate([vec_ro, vec_do, vec_rt, vec_at]).astype(np.float32)
