"""
train_neo.py — Train NeoRacer using OpenAI Evolution Strategies.

Strategy:
  - Antithetic sampling   : each noise vector is evaluated as +eps AND -eps,
                            halving variance for free.
  - Rank normalisation    : scores are mapped to [-0.5, 0.5] before computing
                            the gradient, making learning rate scale-invariant.
  - Waypoint fitness      : the track's reset_vectors are used as ordered
                            checkpoints. Score = total waypoints cleared, which
                            directly rewards lap progress (not just top speed).

Fully headless — no display window required.

Usage:
    python train_neo.py
"""

import sys
import signal
import datetime
import numpy as np
from multiprocessing import Pool
from lib.Map import Map
from players.NeoRacer import NeoNetwork, NeoRacer


def _worker_init():
    """Make worker processes ignore SIGINT so only the main process handles Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ── Hyperparameters ────────────────────────────────────────────────────────────

TICKS_PER_EVAL  = 1500       # ticks per rollout (~50s) — enough for a full lap + ~4 crash resets (2s each)
N_PAIRS         = 24         # antithetic pairs   → 48 rollouts per generation
SIGMA_INIT      = 0.05       # initial ES perturbation std
SIGMA_RESET     = 0.10       # larger sigma used on stagnation reset to escape local optima
SIGMA_DECAY     = 0.9999     # slowly shrink noise as training matures
SIGMA_MIN       = 0.005      # floor — never decay below this
# Adam optimiser hyperparameters for the mean update
ADAM_LR         = 0.01
ADAM_BETA1      = 0.9
ADAM_BETA2      = 0.999
ADAM_EPS        = 1e-8
# Stagnation recovery: if all-time best doesn't improve for this many gens,
# reset sigma to SIGMA_INIT so the search can escape the local optimum.
STAGNATE_GENS   = 30
MAX_GEN         = 10_000
SAVE_EVERY      = 5
WAYPOINT_RADIUS = 50        # pixels — radius to "claim" a waypoint as visited
CRASH_PENALTY   = 40        # waypoints deducted per wall hit (2s reset ≈ 60 ticks ≈ ~40 wps at race speed)
TICK_DT         = 1.0 / 30  # seconds per game tick

# ── Environment setup ──────────────────────────────────────────────────────────

_map    = Map("tracks/1_color.png", "tracks/1_map.png")
_spawns = _map.get_spawns()

try:
    _vectors = np.load("tracks/1_reset_vectors.npy")
    _map.reset_vectors = _vectors
except Exception:
    print("Generating reset vectors …")
    _vectors = _map.genarate_reset_vectors()
    np.save("tracks/1_reset_vectors.npy", _vectors)

_waypoints = _map.reset_vectors[:, :2].astype(np.float64)
_N_WP      = len(_waypoints)
_spawn_pos = np.array(_spawns[0], dtype=np.float32)


# ── Fitness function ───────────────────────────────────────────────────────────

def _evaluate(params: np.ndarray) -> float:
    """
    Simulate one episode and return the waypoint-progress score.
    Waypoints must be visited in track order — the car cannot farm
    score by looping back over already-counted checkpoints.
    """
    net = NeoNetwork()
    net.set_params(params)

    player = NeoRacer(_map, (0, 200, 0), net)
    player.car.position = _spawn_pos.copy()
    player.car.angle    = -np.pi / 2.0
    player.car.velocity = np.array([0.0, 0.0], dtype=np.float32)

    current_wp = 0
    crashes    = 0

    for _ in range(TICKS_PER_EVAL):
        rays = player.cast_rays()
        player.control(rays)
        hit_wall, _ = player.car.update(TICK_DT)

        if hit_wall:
            crashes += 1
            _map.reset_car(player.car)

        # Advance sequential waypoint pointer.
        # Capped at _N_WP iterations per tick to avoid pathological cases.
        for _ in range(_N_WP):
            dist = np.linalg.norm(player.car.position - _waypoints[current_wp % _N_WP])
            if dist < WAYPOINT_RADIUS:
                current_wp += 1
            else:
                break

    return float(current_wp) - crashes * CRASH_PENALTY


# ── Initialise weights, train ─────────────────────────────────────────────────
# Guard required on Windows: without this, each spawned worker re-executes this
# block and tries to spawn its own pool, causing a fork bomb.

if __name__ == '__main__':
    mean_net = NeoNetwork()
    try:
        mean_net.load("players/neo_model.npz")
        print("Resuming from players/neo_model.npz")
    except Exception:
        print("Starting from random weights")

    mean     = mean_net.get_params().astype(np.float64)
    n_params = len(mean)
    sigma    = SIGMA_INIT

    # Adam optimiser state
    adam_m  = np.zeros(n_params)   # 1st moment
    adam_v  = np.zeros(n_params)   # 2nd moment
    adam_t  = 0                    # step counter

    # Stagnation tracking
    gens_without_improvement = 0

    best_score  = _evaluate(mean)
    best_params = mean.copy()

    print(f"Parameters  : {n_params}")
    print(f"Rollouts/gen: {N_PAIRS * 2}  (parallel)")
    print(f"Initial score: {best_score:.1f}  (waypoints cleared in {TICKS_PER_EVAL} ticks)")
    sys.stdout.flush()

    # Persistent pool — workers are spawned once, reused every generation.
    # _worker_init makes workers ignore SIGINT; only the main process catches Ctrl+C.
    with Pool(initializer=_worker_init) as pool:

        # ── ES training loop ──────────────────────────────────────────────────

        try:
            for gen in range(1, MAX_GEN + 1):

                # Build antithetic parameter list and evaluate all in parallel
                noise      = np.random.randn(N_PAIRS, n_params)
                all_params = ([mean + sigma * eps for eps in  noise] +
                              [mean - sigma * eps for eps in  noise])
                all_scores = np.array(pool.map(_evaluate, all_params))
                all_noise  = np.concatenate([noise, -noise])   # shape (2·N_PAIRS, D)

                # Rank normalisation: map scores to [-0.5, 0.5]
                order        = np.argsort(all_scores)
                ranks        = np.empty_like(all_scores)
                ranks[order] = np.arange(len(all_scores), dtype=np.float64)
                normalised   = ranks / (len(ranks) - 1) - 0.5

                # Adam update on the mean
                grad    = (normalised[:, None] * all_noise).mean(axis=0) / sigma
                adam_t += 1
                adam_m  = ADAM_BETA1 * adam_m + (1 - ADAM_BETA1) * grad
                adam_v  = ADAM_BETA2 * adam_v + (1 - ADAM_BETA2) * grad**2
                m_hat   = adam_m / (1 - ADAM_BETA1**adam_t)
                v_hat   = adam_v / (1 - ADAM_BETA2**adam_t)
                mean    = mean + ADAM_LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

                # Track all-time best individual
                best_idx  = int(np.argmax(all_scores))
                candidate = mean + sigma * all_noise[best_idx]
                if all_scores[best_idx] > best_score:
                    best_score  = all_scores[best_idx]
                    best_params = candidate.copy()
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1

                # Sigma schedule: decay normally, but reset when stuck.
                # Also clear Adam moments so momentum doesn't drag back to the old optimum.
                if gens_without_improvement >= STAGNATE_GENS:
                    sigma   = SIGMA_RESET
                    # Kick mean out of the basin Adam converged to
                    mean    = best_params + np.random.randn(n_params) * SIGMA_RESET * 0.5
                    adam_m  = np.zeros(n_params)
                    adam_v  = np.zeros(n_params)
                    adam_t  = 0
                    gens_without_improvement = 0
                    print(f"  [reset at gen {gen}: sigma={SIGMA_RESET}, mean perturbed from best]")
                else:
                    sigma = max(SIGMA_MIN, sigma * SIGMA_DECAY)

                if gen % SAVE_EVERY == 0 or gen == 1:
                    save_net = NeoNetwork()
                    save_net.set_params(best_params)
                    save_net.save("players/neo_model.npz")
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{ts}] Gen {gen:5d} | "
                        f"gen_best={all_scores[best_idx]:.1f} | "
                        f"all_time={best_score:.1f} | "
                        f"mean_score={all_scores.mean():.1f} | "
                        f"sigma={sigma:.5f} | "
                        f"stagnant={gens_without_improvement}/{STAGNATE_GENS}"
                    )
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nInterrupted — saving best weights so far …")

    # ── Final save ────────────────────────────────────────────────────────────

    final_net = NeoNetwork()
    final_net.set_params(best_params)
    final_net.save("players/neo_model.npz")
    print("Training complete. Saved to players/neo_model.npz")
