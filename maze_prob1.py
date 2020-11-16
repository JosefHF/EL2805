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
        #self.transition_probabilities_minotaur = self._transitions(True)
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

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        hitting_maze_obstacle = False
        x_index = 0
        y_index = 1

        # Compute the future position given current (state, action)
        row_player = self.states[state][x_index] + self.actions[action][x_index];
        col_player = self.states[state][y_index] + self.actions[action][y_index];
        # Is the future position an impossible one ?
        hitting_maze_walls = ((row_player == -1) or (row_player == self.maze.shape[x_index]) or (col_player == -1) or (col_player == self.maze.shape[y_index]) or self.maze[row_player,col_player] == 1)

        x_index = 2
        y_index = 3

        # Compute possible minotaur movements
        minotaur_possible_positions = []
        for action_key in self.actions_minotaur.keys():
            row_minotaur = self.states[state][x_index] + self.actions_minotaur[action_key][0];
            col_minotaur = self.states[state][y_index] + self.actions_minotaur[action_key][1];

            next_row_box = row_minotaur + self.actions_minotaur[action_key][0];
            next_col_box = col_minotaur + self.actions_minotaur[action_key][1];
            
            
            

            

            minotaur_possible_walks =   (row_minotaur != -1) and (row_minotaur != self.maze.shape[0]) \
                                        and (col_minotaur != -1) \
                                        and (col_minotaur != self.maze.shape[1])
                                        
            if minotaur_possible_walks:

                minotaur_jump_option = self.maze[row_minotaur,col_minotaur] == 1
                minotaur_hit_obst = next_row_box == self.maze.shape[0] \
                                or next_col_box == self.maze.shape[1] \
                                or next_row_box == -1 \
                                or next_col_box == -1 \
                                or self.maze[next_row_box,next_col_box] == 1

                if minotaur_jump_option:
                    if not minotaur_hit_obst:
                        minotaur_possible_positions.append((next_row_box, next_col_box))
                else:
                    minotaur_possible_positions.append((row_minotaur, col_minotaur))

        
        # minotaur_possible_positions = [(1,1), (1,0)]
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return [self.map[(self.states[state][0], self.states[state][1], mp[0], mp[1])] for mp in minotaur_possible_positions]
        else:
            return [self.map[(row_player, col_player, mp[0], mp[1])] for mp in minotaur_possible_positions]

    def __transitions(self, minotaur = False):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        #if minotaur:
        #    n_actions = self.n_actions_minotaur
        #else:
        n_actions = self.n_actions
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,n_actions);
        transition_probabilities = np.zeros(dimensions);
        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(n_actions):
                next_possible_s = self.__move(s,a);
                for next_s in next_possible_s:
                    #next_s = (next_player_pose[0], next_player_pose[1], minotaur_pose[0], minotaur_pose[1]);
                    transition_probabilities[next_s, s, a] = 1*(1/len(next_possible_s));
        return transition_probabilities;

    def __unchanged_position(self, s, next_s):
        s_state = self.states[s]
        next_s_state = self.states[next_s]

        (s_x, s_y) = (s_state[0],s_state[1])

        (s_n_x, s_n_y) = (next_s_state[0], next_s_state[1])
        
        if (s_x, s_y) == (s_n_x, s_n_y):
            return True
        else:
            return False

    def __caught_by_minotaur(self, next_s):
        state = self.states[next_s]
        if (state[0] == state[2] and state[1] == state[3]):
            return True
        else:
            return False

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a);
                    for potential_state in next_s:
                        # Reward for hitting a wall
                        if self.__unchanged_position(s, potential_state) and a != self.STAY:
                            rewards[s,a] = self.IMPOSSIBLE_REWARD;
                        # Reward for reaching the exit
                        elif self.__unchanged_position(s, potential_state) and self.maze[(self.states[potential_state][0], self.states[potential_state][1])] == 2:
                            if self.__caught_by_minotaur(potential_state):
                                rewards[s,a] = self.IMPOSSIBLE_REWARD
                            else:
                                rewards[s,a] = self.GOAL_REWARD;
                        # Reward for taking a step to an empty cell that is not the exit
                        elif self.__caught_by_minotaur(potential_state):
                            rewards[s,a] = self.IMPOSSIBLE_REWARD
                        elif np.abs(self.states[potential_state][0]-self.states[potential_state][2]) < 2 and np.abs(self.states[potential_state][1]-self.states[potential_state][3]) < 2:
                            rewards[s,a] = -2
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
                     #TODO: Does not work with minotaur, needs 4 indecies.
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];

        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                import random
                next_s = random.choice(next_s)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path

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


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
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


    # Update the color at each frame
    for i in range(len(path)):
        player_path = path[i][0:2]
        mino_path = path[i][2:4]
        
        #Player
        grid.get_celld()[(player_path)].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(player_path)].get_text().set_text('Player')
        
        #Minotaur
        grid.get_celld()[(mino_path)].set_facecolor(LIGHT_RED)
        grid.get_celld()[(mino_path)].get_text().set_text('Minotaur')

        if i > 0:
            
            player_path_prev = path[i-1][0:2]
            
            mino_path_prev = path[i-1][2:4]
            
            if path[i] == path[i-1]:
                grid.get_celld()[(player_path)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(player_path)].get_text().set_text('Player is out')
            else:
                grid.get_celld()[(player_path_prev)].set_facecolor(col_map[maze[player_path_prev]])
                if player_path_prev == mino_path:
                    grid.get_celld()[(player_path_prev)].get_text().set_text('Player is caught')
                else:
                    grid.get_celld()[(player_path_prev)].get_text().set_text('')

                grid.get_celld()[(mino_path_prev)].set_facecolor(col_map[maze[mino_path_prev]])
                if player_path_prev == mino_path:
                    grid.get_celld()[(mino_path_prev)].get_text().set_text('Minotaur won')
                else:
                    grid.get_celld()[(mino_path_prev)].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])



'''env = Maze(maze)
V, policy = dynamic_programming(env, 20)
print("V: ", V, "Policy: ", policy)'''