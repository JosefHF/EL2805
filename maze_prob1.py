import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.actions_minotaur         = self.__actions(True);
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_actions_minotaur       = len(self.actions_minotaur);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.transition_probabilities_minotaur = self._transitions(True)
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

    def __actions(self, minotaur = False):
        actions = dict();
        if not minotaur:
            actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for px in range(self.maze.shape[0]):
            for py in range(self.maze.shape[1]):
                if self.maze[px,py] != 1:
                    for mx in range(self.maze.shape[0]):
                        for my in range(self.maze.shape[1]):
                            states[s] = (px,py,mx,my);
                            map[(px,py,mx,my)] = s;
                            s += 1;
        return states, map

    def __move(self, state, action, minotaur=False):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        hitting_maze_obstacle = False
        x_index = 0
        y_index = 1
        if minotaur:
            x_index = 2
            y_index = 3

        # Compute the future position given current (state, action)
        row = self.states[state][x_index] + self.actions[action][x_index];
        col = self.states[state][y_index] + self.actions[action][y_index];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[x_index]) or \
                              (col == -1) or (col == self.maze.shape[y_index])

        if not minotaur:
            hitting_maze_obstacle = self.maze[row,col] == 1
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls or hitting_maze_obstacle:
            return state;
        else:
            return self.map[(row, col)];

    def __transitions(self, minotaur = False):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        if minotaur:
            n_actions = self.n_actions_minotaur
        else:
            n_actions = self.n_actions
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,n_actions);
        transition_probabilities = np.zeros(dimensions);
        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(n_actions):
                next_s = self.__move(s,a,minotaur);
                transition_probabilities[next_s, s, a] = 1;
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a);
                    # Rewrd for hitting a wall
                    if s == next_s and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD;
                    # Reward for reaching the exit
                    elif s == next_s and self.maze[self.states[next_s]] == 2:
                        rewards[s,a] = self.GOAL_REWARD;
                    # Reward for taking a step to an empty cell that is not the exit
                    elif next_s[0] == next_s[2] and next_s[1] == next_s[3]:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD
                    else:
                        rewards[s,a] = self.STEP_REWARD;
                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]]<0:
                        row, col = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];

        return rewards;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);
