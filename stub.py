# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
# from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900
VEL_BINS = 3


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self, gamma, learn, expl):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gamma = gamma
        self.learning = learn
        self.exploration = expl
        self.score = 0

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE, VEL_BINS + 1, Y_SCREEN // Y_BINSIZE))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        vel = int(VEL_BINS/(1 + np.exp(state["monkey"]["vel"])))
        y_top = int((Y_SCREEN - state["monkey"]["top"]) // Y_BINSIZE)
        score = state["score"]
        return (rel_x, rel_y, vel, y_top, score)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # TODO (currently monkey just jumps around randomly)
        # 1. Discretize 'state' to get your transformed 'current state' features.
        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        # 3. Choose the next action using an epsilon-greedy policy.


        if self.last_state is None:
            new_action = int(npr.rand() < 0.1)
        else:
            current_state = self.discretize_state(state)
            last_state = self.discretize_state(self.last_state)

            self.exploration = np.exp(-1*self.score - 9)

            self.Q[self.last_action][last_state[0]][last_state[1]][last_state[2]][last_state[3]] += self.learning * (self.last_reward + self.gamma * self.best_reward(current_state) - self.Q[self.last_action][last_state[0]][last_state[1]][last_state[2]][last_state[3]])
            if self.Q[1][current_state[0]][current_state[1]][current_state[2]][current_state[3]] >= self.Q[0][current_state[0]][current_state[1]][current_state[2]][current_state[3]]:
                best_move = 1
            else:
                best_move = 0

            new_action = best_move if npr.rand() < 1 - self.exploration else npr.choice([0, 1])


        new_state = state

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def best_reward(self, state):
        return max([self.Q[i][state[0]][state[1]][state[2]][state[3]] for i in [0, 1]])

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner(gamma=0.825, expl=.00001, learn=0.005*(11))

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)
    print(hist)
    print(max(hist))  

    # Save history. 
    np.save('hist', np.array(hist))

    # vals = []
    # for i, disc in enumerate([0.825]):
    #     vals.append([])
    #     for j, learn in enumerate([0.005*i for i in range(20)]):
    #         vals[i].append([])
    #         for k, expl in enumerate(
    #             [.001]
    #         ):

    #             hist = []
    #             s = 0

    #             for _ in range(5):
    #                 agent = Learner(gamma=disc, learn=learn, expl=expl)
    #                 hist = []
    #                 run_games(agent, hist, 100, 100)
    #                 s += np.mean(hist[80:])
    #             vals[i][j].append(s / 5)

    # np.save("vals", np.array(vals))
    # print(vals)