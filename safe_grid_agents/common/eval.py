"""Agent-specific evaluation interactions."""
import numpy as np

from . import utils as ut
from collections import defaultdict


def default_eval(agent, env, eval_history, args):
    """Evaluate an agent (default interaction)."""
    print("#### EVAL ####")
    eval_over = False
    t = 0
    (step_type, reward, discount, state), done = env.reset(), False
    board = state["board"]

    show = args.eval_visualize_episodes > 0
    next_animation = [np.copy(state["RGB"])]
    episodes_to_show = []

    while True:
        if done:
            eval_history = ut.track_metrics(eval_history, env, eval=True, write=False)
            (step_type, reward, discount, state), done = env.reset(), False
            board = state["board"]
            if show:
                animation = np.stack(next_animation)
                animation = np.swapaxes(animation, 0, 1)  # swap color and time axes
                episodes_to_show.append(animation)
                next_animation = [np.copy(state["RGB"])]
                show = args.eval_visualize_episodes > len(episodes_to_show)
            if eval_over:
                break

        action = agent.act(board)
        step_type, reward, discount, successor = env.step(action)
        board = successor["board"]

        done = step_type.value == 2
        t += 1
        eval_over = t >= args.eval_timesteps

        if show:
            next_animation.append(np.copy(successor["RGB"]))

    if len(episodes_to_show) > 0:
        animation_tensor = np.stack(episodes_to_show)
        eval_history["writer"].add_video(
            "Evaluation/grid_animation", animation_tensor, eval_history["period"]
        )

    eval_history = ut.track_metrics(eval_history, env, eval=True)
    eval_history["returns"].reset(reset_history=True)
    eval_history["safeties"].reset()
    eval_history["margins"].reset()
    eval_history["margins_support"].reset()
    eval_history["period"] += 1
    return eval_history


eval_map = defaultdict(lambda: default_eval, {})
