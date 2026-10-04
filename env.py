import numpy as np


class GridEnvCells:
    def __init__(self, num_static=7, num_dynamic=1, seed=None):
        self.W = self.H = 40  # Grid size normal 30, test 20
        self.min_cell = np.array([0, 0], dtype=int)
        self.max_cell = np.array([self.W - 1, self.H - 1], dtype=int)

        self.num_static = int(num_static)
        self.num_dynamic = int(num_dynamic)
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Start/terminal cells (fixed as in paper)
        self.start_cell = np.array([1, 1], dtype=int)
        self.terminal_cell = np.array([29, 29], dtype=int)

        # Robot heading (float, radians)
        self.heading = 0.0

        # Paper's 5 actions
        # Left, Forward-Left, Forward, Forward-Right, Right
        # Relative to current heading
        self.action_names = ["Left", "Forward-Left", "Forward", "Forward-Right", "Right"]
        self.action_yaw_deltas = np.deg2rad([-90, -45, 0, 45, 90])

        # Sensor parameters
        self.sensor_radius = 5.0  # cells

        # Dynamic obstacle cardinal directions (for movement)
        self.cardinal_dirs = np.array([
            [0, 1],   
            [1, 0],   
            [0, -1],  
            [-1, 0],  
        ], dtype=int)

        # Termination/pacing
        # 300 should be more than enough
        self.max_steps = 300
        self.steps = 0

        # Penalties
        self.collision_penalty = 100.0
        self.dyn_move_toggle = False

        # Initialize obstacle arrays
        self.static_obstacles = np.zeros((0, 2), dtype=int)
        self.dynamic_obstacles = np.zeros((0, 2), dtype=int)
        self.dynamic_dirs_idx = np.array([], dtype=int)
        self.pos = self.start_cell.copy()

        self.reset()

    def _sample_cells(self, n, forbidden):
        """
        Sample n unique grid cells (ints), avoiding 'forbidden' (set of tuples).
        Returns cells in the order they were sampled for reproducibility.
        """
        chosen = []
        tries = 0
        max_tries = 20000
        while len(chosen) < n and tries < max_tries:
            x = self.rng.integers(self.min_cell[0], self.max_cell[0] + 1)
            y = self.rng.integers(self.min_cell[1], self.max_cell[1] + 1)
            c = (int(x), int(y))
            if c not in forbidden and c not in chosen:
                chosen.append(c)
            tries += 1
        if len(chosen) < n:
            raise RuntimeError("Could not sample enough obstacle cells without overlap.")
        return np.array(chosen, dtype=int)

    @staticmethod
    def _wrap_to_pi(theta):
        """Wrap angle to [-π, π]."""
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def _region_index(self, vec):
        """
        Map vector (dx,dy) to region R1..R8 as defined in paper (Fig 5).
        Using WORLD-FIXED coordinates (not relative to agent heading).
        R1: N, R2: NE, R3: E, R4: SE, R5: S, R6: SW, R7: W, R8: NW
        """
        dx, dy = vec
        if dx == 0 and dy == 0:
            return 1
        angle_deg = np.degrees(np.arctan2(dy, dx))

        if 67.5 <= angle_deg < 112.5:
            return 1  # North
        elif 22.5 <= angle_deg < 67.5:
            return 2  # NE
        elif -22.5 <= angle_deg < 22.5:
            return 3  # East
        elif -67.5 <= angle_deg < -22.5:
            return 4  # SE
        elif -112.5 <= angle_deg < -67.5:
            return 5  # South
        elif -157.5 <= angle_deg < -112.5:
            return 6  # SW
        elif angle_deg >= 157.5 or angle_deg < -157.5:
            return 7  # West
        elif 112.5 <= angle_deg < 157.5:
            return 8  # NW
        else:
            return 1

    def _angle_between_vectors(self, v_goal, v_obs):
        """
        Calculate angle α (alpha) between target line and obstacle line.
        Paper: Equation (22), α ∈ [0, π]
        Returns bin index 1..8
        """
        a = np.array(v_goal, dtype=float)
        b = np.array(v_obs, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-9 or norm_b < 1e-9:
            return 1

        a /= norm_a
        b /= norm_b

        cosang = float(np.clip(a @ b, -1.0, 1.0))
        alpha = np.arccos(cosang)  # [0, π]

        bin_w = np.pi / 8
        idx = min(int(alpha // bin_w), 7)
        return idx + 1

    def _in_bounds(self, cell):
        """Check if cell is within grid bounds."""
        return (self.min_cell <= cell).all() and (cell <= self.max_cell).all()

    def _is_obstacle_at(self, cell, exclude_dynamic_idx=None):
        """Check if a cell contains an obstacle."""
        # Check static obstacles
        if len(self.static_obstacles) > 0:
            if any((cell == self.static_obstacles).all(axis=1)):
                return True

        # Check dynamic obstacles
        if len(self.dynamic_obstacles) > 0:
            for i in range(len(self.dynamic_obstacles)):
                if exclude_dynamic_idx is not None and i == exclude_dynamic_idx:
                    continue
                if (cell == self.dynamic_obstacles[i]).all():
                    return True

        return False

    def _region_index_relative(self, vec):
        """
        Map vec to R1..R8 *relative* to current heading.
        R3 ~ "straight ahead", R1/R5 "left/right rear", etc.
        """
        dx, dy = vec
        if dx == 0 and dy == 0:
            return 1

        angle_world = np.arctan2(dy, dx)
        rel_angle = self._wrap_to_pi(angle_world - self.heading)
        angle_deg = np.degrees(rel_angle)

        if 67.5 <= angle_deg < 112.5:
            return 1
        elif 22.5 <= angle_deg < 67.5:
            return 2
        elif -22.5 <= angle_deg < 22.5:
            return 3  # straight ahead
        elif -67.5 <= angle_deg < -22.5:
            return 4
        elif -112.5 <= angle_deg < -67.5:
            return 5
        elif -157.5 <= angle_deg < -112.5:
            return 6
        elif angle_deg >= 157.5 or angle_deg < -157.5:
            return 7
        elif 112.5 <= angle_deg < 157.5:
            return 8
        else:
            return 1

    def reset(self, seed=None):
        """
        Reset environment to initial state.

        Args:
            seed: Optional seed to override the environment's seed for this reset.
                  If None, uses the environment's original seed.
        """
        # Handle seeding
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)
        elif self._seed is not None:
            self.rng = np.random.default_rng(self._seed)

        # Define forbidden locations
        forbidden = {
            tuple(self.start_cell),
            tuple(self.terminal_cell),
        }

        # Sample static obstacles
        if self.num_static > 0:
            self.static_obstacles = self._sample_cells(self.num_static, forbidden)
            forbidden |= set(map(tuple, self.static_obstacles))
            # 50% chance to have static obstacle in (n,n) blocking diogonal
            # let n  be random number from 1 to 27
            n = self.rng.integers(1, 28)
            diag_cell = (self.start_cell[0] + n, self.start_cell[1] + n)
            if diag_cell not in forbidden and self.rng.random() < 0.9:
                self.static_obstacles = np.vstack([self.static_obstacles, diag_cell])
                forbidden.add(diag_cell)
        else:
            self.static_obstacles = np.zeros((0, 2), dtype=int)

        # Sample dynamic obstacles
        if self.num_dynamic > 0:
            self.dynamic_obstacles = self._sample_cells(self.num_dynamic, forbidden)
            self.dynamic_dirs_idx = self.rng.integers(0, 4, size=self.num_dynamic)
        else:
            self.dynamic_obstacles = np.zeros((0, 2), dtype=int)
            self.dynamic_dirs_idx = np.array([], dtype=int)

        # Reset agent
        self.pos = self.start_cell.copy()
        self.heading = 0.0
        self.steps = 0
        self.dyn_move_toggle = False

        return self._get_state()

    def _move_dynamics(self):
        """
        Move dynamic obstacles by one cell along their cardinal direction;
        bounce off walls and obstacles.
        """
        if self.num_dynamic == 0:
            return

        new_positions = []
        for i in range(self.num_dynamic):
            current_pos = self.dynamic_obstacles[i].copy()
            step = self.cardinal_dirs[self.dynamic_dirs_idx[i]]
            nxt = current_pos + step
            move_blocked = False

            # Check bounds
            if not self._in_bounds(nxt):
                move_blocked = True
            # Check static obstacles
            elif len(self.static_obstacles) > 0 and any((nxt == self.static_obstacles).all(axis=1)):
                move_blocked = True
            # Check agent position
            elif (nxt == self.pos).all():
                move_blocked = True
            # Check other dynamic obstacles (already moved)
            elif len(new_positions) > 0:
                for prev_pos in new_positions:
                    if (nxt == prev_pos).all():
                        move_blocked = True
                        break

            # Check other dynamic obstacles (not yet moved)
            if not move_blocked:
                for j in range(i + 1, self.num_dynamic):
                    if (nxt == self.dynamic_obstacles[j]).all():
                        move_blocked = True
                        break

            # If blocked, reverse direction
            if move_blocked:
                # Reverse: 0<->2 (up<->down), 1<->3 (right<->left)
                self.dynamic_dirs_idx[i] = (self.dynamic_dirs_idx[i] + 2) % 4
                nxt = current_pos

            new_positions.append(nxt)

        # Apply all new positions
        for i, new_pos in enumerate(new_positions):
            self.dynamic_obstacles[i] = new_pos

    def _nearest_obstacle(self):
        """
        Return (pos, is_dynamic, dyn_dir_idx or None, dist_L2)
        """
        all_pos = []
        tags = []

        for p in self.static_obstacles:
            all_pos.append(p)
            tags.append(("static", None))

        for i, p in enumerate(self.dynamic_obstacles):
            all_pos.append(p)
            tags.append(("dynamic", int(self.dynamic_dirs_idx[i])))

        if len(all_pos) == 0:
            return self.pos.copy(), False, None, 0.0

        all_pos = np.array(all_pos, dtype=int)
        dists = np.linalg.norm(all_pos - self.pos, axis=1)
        i = int(np.argmin(dists))
        kind, dir_idx = tags[i]
        pos = all_pos[i]
        is_dyn = (kind == "dynamic")
        dist = float(dists[i])

        # Only consider dynamic if within sensor range
        if is_dyn and dist > self.sensor_radius:
            is_dyn = False
            dir_idx = None

        return pos, is_dyn, dir_idx, dist

    def seed(self, seed=None):
        """
        Set the seed for the environment's random number generator.
        Note: This only affects future resets, not current state.

        Args:
            seed: The seed value. If None, uses system entropy.

        Returns:
            The seed that was set.
        """
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        return seed

    def _get_state(self):
        """
        Paper's state representation (Equation 23):
        S_T = [R_o, D_o, R_T, A_T→o]
        - R_o: Region of nearest obstacle (1..8)
        - D_o: Direction of dynamic obstacle (1..4) or 0 if static/out of range
        - R_T: Region of goal (1..8)
        - A_T→o: Angle bin between target line and obstacle line (1..8)
        """
        goal_vec = self.terminal_cell - self.pos
        obs_pos, is_dyn, dir_idx, _ = self._nearest_obstacle()
        obs_vec = obs_pos - self.pos

        R_T = self._region_index_relative(goal_vec)
        R_o = self._region_index_relative(obs_vec)
        A_T_o = self._angle_between_vectors(goal_vec, obs_vec)

        # D_o: Dynamic obstacle direction (1-4) or 0 if static
        D_o = (int(dir_idx) + 1) if is_dyn else 0

        return np.array([R_o, D_o, R_T, A_T_o], dtype=int)

    def _heading_to_step_vector(self, heading):
        """Convert heading (radians) to grid step vector."""
        heading_deg = np.degrees(heading)
        if heading_deg < 0:
            heading_deg += 360

        if 337.5 <= heading_deg or heading_deg < 22.5:
            return np.array([1, 0])   # East
        elif 22.5 <= heading_deg < 67.5:
            return np.array([1, 1])   # NE
        elif 67.5 <= heading_deg < 112.5:
            return np.array([0, 1])   # North
        elif 112.5 <= heading_deg < 157.5:
            return np.array([-1, 1])  # NW
        elif 157.5 <= heading_deg < 202.5:
            return np.array([-1, 0])  # West
        elif 202.5 <= heading_deg < 247.5:
            return np.array([-1, -1]) # SW
        elif 247.5 <= heading_deg < 292.5:
            return np.array([0, -1])  # South
        else:
            return np.array([1, -1])  # SE

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: Integer 0-4 corresponding to action_names

        Returns:
            state, reward, done, info
        """
        assert 0 <= action <= 4, f"action must be in [0,4], got {action}"

        prev_pos = self.pos.copy()
        info = {}
        done = False

        # 1) Update heading
        self.heading = self._wrap_to_pi(self.heading + self.action_yaw_deltas[action])

        # 2) Compute intended step vector
        step_vec = self._heading_to_step_vector(self.heading)
        nxt = self.pos + step_vec

        # 3) Check for collisions - END EPISODE ON ANY COLLISION
        out_of_bounds = not self._in_bounds(nxt)
        hit_static = len(self.static_obstacles) > 0 and any((nxt == self.static_obstacles).all(axis=1))
        hit_dynamic = len(self.dynamic_obstacles) > 0 and any((nxt == self.dynamic_obstacles).all(axis=1))

        if out_of_bounds:
            info["out_of_bounds"] = True
            reward = -self.collision_penalty
            done = True
            self.steps += 1
            return self._get_state(), float(reward), done, info

        if hit_static:
            info["collision"] = True
            info["collision_type"] = "static"
            reward = -self.collision_penalty
            done = True
            self.steps += 1
            return self._get_state(), float(reward), done, info

        if hit_dynamic:
            info["collision"] = True
            info["collision_type"] = "dynamic"
            reward = -self.collision_penalty
            done = True
            self.steps += 1
            return self._get_state(), float(reward), done, info

        # 4) Valid move - update position
        self.pos = nxt

        # 5) Move dynamic obstacles (every other step)
        if self.dyn_move_toggle:
            self._move_dynamics()
        self.dyn_move_toggle = not self.dyn_move_toggle

        # 6) Check if dynamic obstacle moved INTO agent
        if len(self.dynamic_obstacles) > 0:
            for obs in self.dynamic_obstacles:
                if (self.pos == obs).all():
                    info["collision"] = True
                    info["collision_type"] = "dynamic_into_agent"
                    reward = -self.collision_penalty
                    done = True
                    self.steps += 1
                    return self._get_state(), float(reward), done, info

        # 7) Compute reward (only if no collision)
        reward = self._reward2(prev_pos)

        self.steps += 1

        # 8) Check goal
        if (self.pos == self.terminal_cell).all():
            info["goal"] = True
            reward += self.collision_penalty
            done = True
            return self._get_state(), float(reward), done, info

        # 9) Timeout
        if self.steps >= self.max_steps:
            info["timeout"] = True
            done = True
            reward -= self.collision_penalty 
        
        # Used to penalize long trajectories
        reward -= 0.1

        return self._get_state(), float(reward), done, info
    
    def _nearest_obstacle_from(self, pos):
        all_pos = []
        tags = []
        for p in self.static_obstacles:
            all_pos.append(p); tags.append(("static", None))
        for i, p in enumerate(self.dynamic_obstacles):
            all_pos.append(p); tags.append(("dynamic", int(self.dynamic_dirs_idx[i])))
    
        if len(all_pos) == 0:
            return pos, False, None, 0.0
    
        all_pos = np.array(all_pos, dtype=int)
        dists = np.linalg.norm(all_pos - pos, axis=1)
        i = int(np.argmin(dists))
        kind, dir_idx = tags[i]
        pos_i = all_pos[i]
        is_dyn = (kind == "dynamic")
        return pos_i, is_dyn, dir_idx, float(dists[i])

    def _reward2(self, prev_pos):
        """
        DSQN reward (Equation 24 from paper):
            r = β1 * (d_prev - d_cur) + e^{β2 * (d_obs_prev - d_obs_cur)} + β3 * sin(α)
        """
        # 1) Goal distance progress
        d_prev = np.linalg.norm(self.terminal_cell - prev_pos)
        d_cur = np.linalg.norm(self.terminal_cell - self.pos)
        goal_progress = d_prev - d_cur  # >0 when closer to goal

        # 2) Obstacle distance (nearest) - EXPONENTIAL term per paper
        _, _, _, d_obs_prev = self._nearest_obstacle_from(prev_pos)
        _, _, _, d_obs_cur = self._nearest_obstacle_from(self.pos)
        obs_delta = d_obs_prev - d_obs_cur  # >0 when CLOSER to obstacle (bad)
        
        # 3) Angle term sin(α) between goal and obstacle vectors
        goal_vec = self.terminal_cell - self.pos
        obs_pos, _, _, _ = self._nearest_obstacle()
        obs_vec = obs_pos - self.pos

        norm_g = np.linalg.norm(goal_vec)
        norm_o = np.linalg.norm(obs_vec)
        if norm_g > 1e-6 and norm_o > 1e-6:
            cos_alpha = np.clip(np.dot(goal_vec, obs_vec) / (norm_g * norm_o), -1.0, 1.0)
            alpha = np.arccos(cos_alpha)
            angle_term = np.sin(alpha)
        else:
            angle_term = 0.0

        # Paper's coefficients (tune these)
        beta1 = 1.0
        beta2 = -0.5
        beta3 = 0.3

        reward = (
            beta1 * goal_progress +
            (np.exp(beta2 * obs_delta)) +  # Exponential per Eq. 24
            beta3 * angle_term
        )
        return float(reward)
    
    def get_valid_actions(self):
        """Return mask of valid actions that don't go out of bounds or hit obstacles."""
        valid = np.ones(5, dtype=bool)
        for action in range(5):
            test_heading = self._wrap_to_pi(self.heading + self.action_yaw_deltas[action])
            step_vec = self._heading_to_step_vector(test_heading)
            nxt = self.pos + step_vec
            if not self._in_bounds(nxt):
                valid[action] = False
        return valid
