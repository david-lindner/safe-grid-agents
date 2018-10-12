"""PPO Agent for CRMDPs."""
import numpy as np
from typing import Generator

from .policy import PPOAgent
from ...types import Rollout


class PPOCRMDPAgent(PPOAgent):
    """PPO Agent for CRMDPs."""

    def __init__(self, env, args) -> None:
        super().__init__()
        self.corrupt_states = dict()
        self.safe_states = dict()
        self.d = lambda x, y: 3
        self.epsilon = 1e-3
        self.rllb = dict()

    def _mark_state_corrupt(self, board, reward) -> None:
        board_str = board.tostring()
        if board_str in self.non_corrupt_states:
            self.non_corrupt_states.remove(board_str)
        self.corrupt_states[board_str] = reward

    def _mark_state_safe(self, board, reward) -> None:
        self.safe_states[board.tostring()] = reward

    def _is_state_corrupt(self, board) -> bool:
        return board.tostring() in self.corrupt_states

    def _iterate_safe_states(self) -> None:
        for board_str, reward in self.safe_states:
            yield np.fromstring(board_str), reward

    def _iterate_corrupt_states(self) -> Generator[np.array, None, None]:
        for board_str, reward in self.corrupt_states:
            yield np.fromstring(board_str), reward

    def _update_rllb(self) -> None:
        """Update the reward lower Lipschitz bound."""
        iterator_safe = self._iterate_safe_states()
        iterator_corrupt = self._iterate_corrupt_states()

        for corrupt_board, corrupt_reward in iterator_corrupt:
            rllb = None
            for safe_board, safe_reward in iterator_safe:
                bound = safe_reward - self.d(safe_board, corrupt_board)
                if rllb is None or bound > rllb:
                    rllb = bound
            self.rllb[corrupt_board] = rllb

    def _get_TLV(self, boardX, rewardX, state_iterator) -> float:
        """Return the total Lipschitz violation of a state X w.r.t a set of states."""
        TLV = 0
        for boardY, rewardY in state_iterator:
            TLV += abs(rewardX - rewardY) - self.d(boardY, boardX)
        return TLV

    def get_modified_reward(self, board, reward) -> float:
        """Return the reward to use for optimizing the policy based on the rllb."""
        if self._is_state_corrupt(board):
            return self.rllb[board]
        else:
            return reward

    def identify_corruption_in_trajectory(self, boards, rewards) -> None:
        """Perform detection of corrupt states on a trajectory.

        Updates the set of safe states and corrupt states with all new states,
        that are being visited in this trajectory. Then updates the self.rllb
        dict, so that we can get the modified reward function.
        """
        for board, reward in zip(boards, rewards):
            if not self._is_state_corrupt(board):
                self._mark_state_safe(board, reward)

        TLV = np.zeros(len(boards))
        for i in range(len(boards)):
            TLV[i] = self._get_TLV(boards[i], reward[i], self._iterate_safe_states())

        TLV_sort_idx = np.argsort(TLV)[::-1]

        # iterate over all states in the trajectory in order decreasing by their TLV
        for i in range(len(boards)):
            idx = TLV_sort_idx[i]
            new_TLV = self._get_TLV(
                boards[idx], reward[idx], self._iterate_safe_states()
            )
            if new_TLV <= self.epsilon:
                break
            else:
                self._mark_state_corrupt(boards[idx], rewards[idx])
        self._update_rllb()

    def gather_rollout(self, env, env_state, history, args) -> Rollout:
        # TODO: during rollout generation call self.identify_corruption_in_trajectory
        # TODO: then replace reward with self.get_modified_reward
        super().gather_rollout(env, env_state, history, args)
