from __future__ import annotations


class PIDLagrangian:
    """PID-controlled dual price for a per-episode emissions budget constraint.

    Replaces the source paper's unpublished, fixed reward weights (w1..w5)
    with a carbon multiplier that adapts online to keep cumulative emissions
    near the episode's prorated quota (weakness #6).
    """

    def __init__(self, kp: float = 0.02, ki: float = 0.002, kd: float = 0.0, lambda_max: float = 5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lambda_max = lambda_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.value = 0.0

    def update(self, error: float) -> float:
        self.integral = max(0.0, self.integral + error)
        derivative = error - self.prev_error
        self.prev_error = error
        self.value = min(self.lambda_max, max(0.0, self.kp * error + self.ki * self.integral + self.kd * derivative))
        return self.value

    def reset_episode(self) -> None:
        self.prev_error = 0.0
