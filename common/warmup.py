import sys
sys.path.append('../')
import ssrl.warmup as ss
import common.utils as ut
from common.agents import RandomAgent

def dqn_warmup(agent, env, history, args):
    """Warm start for DQN agent"""
    rando = RandomAgent(env, args) # Exploration only
    print("#### WARMUP ####\n")
    done = True

    for i in range(args.replay_capacity):
        if done:
            history['returns'].update(env.episode_return)
            (step_type, reward, discount, state), done = env.reset(), False

        action = rando.act(None)
        step_type, reward, discount, successor = env.step(action)
        done = step_type.value == 2
        agent.replay.add(state, action, reward, successor)

    return agent, env, history, args

def noop(agent, env, history, args):
    return agent, env, history, args

warmup_map = {
    'random':noop,
    'single':noop,
    'tabular-q':noop,
    'deep-q':dqn_warmup,
    'tabular-ssq':ss.random_warmup,
}