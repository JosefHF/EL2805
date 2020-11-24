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
    STEP_REWARD = 0.0
    GOAL_REWARD = 10.0
    IMPOSSIBLE_REWARD = -50.0


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.actions_police           = self.__actions(True);
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_actions_police         = len(self.actions_police);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
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
        # TODO: Add a killed state if eaten

        states = dict();
        map = dict();
        end = False;
        s = 0;
        for px in range(self.maze.shape[0]):
            for py in range(self.maze.shape[1]):
                for mx in range(self.maze.shape[0]):
                    for my in range(self.maze.shape[1]):
                        states[s] = (px,py,mx,my);
                        map[(px,py,mx,my)] = s;
                        s += 1;
        #Killing state
        states[s] = (-1,-1,-1,-1)
        map[(-1,-1,-1,-1)] = s;


        return states, map

    def sameCol(self, state):
        return state[1] == state[3]


    def sameRow(self, state):
        return state[0] == state[2]
    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        #if self.states[state] == (-1,-1,-1,-1):
            #return [self.map[(0,0,1,2)]]

        ## Added for killing state
        if self.states[state][0] == self.states[state][2] and self.states[state][1] == self.states[state][3]:
            return [self.map[(-1,-1,-1,-1)]]

        x_index = 0
        y_index = 1

        # Compute the future position given current (state, action)
        row_player = self.states[state][x_index] + self.actions[action][x_index];
        col_player = self.states[state][y_index] + self.actions[action][y_index];

        # Is the future position an impossible one ?
        hitting_maze_walls = ((row_player == -1) or (row_player == self.maze.shape[x_index]) or (col_player == -1) or (col_player == self.maze.shape[y_index]))

        x_index = 2
        y_index = 3

        prev_distance_row = np.abs(self.states[state][0]-self.states[state][2])
        prev_distance_col = np.abs(self.states[state][1]-self.states[state][3])

        # Compute possible minotaur movements
        police_possible_positions = []
        for action_key in self.actions_police.keys():
            row_police = self.states[state][x_index] + self.actions_police[action_key][0];
            col_police = self.states[state][y_index] + self.actions_police[action_key][1];

            police_possible_walks =   (row_police != -1) and (row_police != self.maze.shape[0]) \
                                        and (col_police != -1) \
                                        and (col_police != self.maze.shape[1])
            if police_possible_walks:
                

                if self.sameCol(self.states[state]):
                    if np.abs(self.states[state][0]-row_police) <= prev_distance_row:
                        police_possible_positions.append((row_police, col_police))
                elif self.sameRow(self.states[state]):
                    if np.abs(self.states[state][1]-col_police) <= prev_distance_col:
                        police_possible_positions.append((row_police, col_police))
                else:
                    if (np.abs(self.states[state][0]-row_police) < prev_distance_row and np.abs(self.states[state][1]-col_police) == prev_distance_col)\
                        or (np.abs(self.states[state][0]-row_police) == prev_distance_row and np.abs(self.states[state][1]-col_police) < prev_distance_col):
                        police_possible_positions.append((row_police, col_police))
                    #print("state: ", self.states[state])
                    #print("row_p", row_police)
                    #print("col_p: ", col_police)
                    #print("prev_distance_row: ", prev_distance_row, " vs ", np.abs(self.states[state][0]-row_police))
                    #print("prev_distance_col: ", prev_distance_col, " vs ", np.abs(self.states[state][1]-col_police))
                    #print("len possible pos: ", len(police_possible_positions))

        if not hitting_maze_walls:
            return [self.map[(row_player, col_player, pp[0],pp[1])] if row_player != pp[0] or col_player != pp[1] else self.map[(-1,-1,-1,-1)] for pp in police_possible_positions]
        else:
            return [self.map[(self.states[state][0], self.states[state][1], pp[0],pp[1])] if self.states[state][0] != pp[0] or self.states[state][1] != pp[1] else self.map[(-1,-1,-1,-1)] for pp in police_possible_positions]

        
        


    def __transitions(self, minotaur = False):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        #if minotaur:
        #    n_actions = self.n_actions_police      
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
                    transition_probabilities[next_s, s, a] = (1.0/float(len(next_possible_s)));
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
                    cummulative_reward = 0.0
                    for potential_state in next_s:
                        if s == self.map[(-1,-1,-1,-1)]:
                            cummulative_reward += 0
                        if self.__caught_by_minotaur(potential_state):
                            cummulative_reward += self.IMPOSSIBLE_REWARD
                        elif self.__unchanged_position(s, potential_state) and a != self.STAY:
                            cummulative_reward += self.IMPOSSIBLE_REWARD
                        elif self.states[potential_state][:2] == (0,0) or (0,5) or (2,0) or (2,5):
                            #print("We have won")
                            #if self.__unchanged_position(s, potential_state):
                                #cummulative_reward += 0.0
                            #else:
                            cummulative_reward += self.GOAL_REWARD
                        else:
                            cummulative_reward += self.STEP_REWARD
                    if len(next_s) == 0:
                        print(self.states[s], " and acvtion ", a)
                    rewards[s,a] = cummulative_reward/len(next_s)
                        
                        
                    '''if random_rewards and self.maze[self.states[next_s]]<0:
                        row, col = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;'''
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
                #print(s, t, next_s)
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

            #Added random choice of minotaurs next action
            import random
            next_s = random.choice(next_s);
            #print(next_s)
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s and t < 1000:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                next_s = random.choice(next_s)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                if next_s == (-1,-1,-1,-1):
                    return path
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

        if i > 0:
            player_path_prev = path[i-1][0:2]
            mino_path_prev = path[i-1][2:4]

            if player_path == mino_path:
                continue
                #grid.get_celld()[player_path_prev].set_facecolor(WHITE)
                #grid.get_celld()[(player_path_prev)].get_text().set_text('')

                #grid.get_celld()[mino_path_prev].set_facecolor(WHITE)
                #grid.get_celld()[(mino_path_prev)].get_text().set_text('')

                #grid.get_celld()[player_path].set_facecolor(LIGHT_RED)
                #grid.get_celld()[(player_path)].get_text().set_text('Player is caught')

            else:
                grid.get_celld()[player_path_prev].set_facecolor(WHITE)
                grid.get_celld()[(player_path_prev)].get_text().set_text('')

                grid.get_celld()[mino_path_prev].set_facecolor(WHITE)
                grid.get_celld()[(mino_path_prev)].get_text().set_text('')

                grid.get_celld()[(player_path)].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(mino_path)].set_facecolor(LIGHT_RED)

                if path[i][0:2] == (6,5):
                    grid.get_celld()[(player_path)].set_facecolor(LIGHT_GREEN)
                    grid.get_celld()[(player_path)].get_text().set_text('Player is out')
                    grid.get_celld()[(mino_path)].get_text().set_text('Minotaur')
                else:
                    grid.get_celld()[(player_path)].get_text().set_text('Player')
                    grid.get_celld()[(mino_path)].get_text().set_text('Minotaur')
        else:
            #Player
            grid.get_celld()[(player_path)].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(player_path)].get_text().set_text('Player')
            
            #Minotaur
            grid.get_celld()[(mino_path)].set_facecolor(LIGHT_RED)
            grid.get_celld()[(mino_path)].get_text().set_text('Minotaur')

        # if i > 0:
            
        #     player_path_prev = path[i-1][0:2]
            
        #     mino_path_prev = path[i-1][2:4]
            
        #     if path[i][0:2] == (6,5):
        #         grid.get_celld()[(player_path)].set_facecolor(LIGHT_GREEN)
        #         grid.get_celld()[(player_path)].get_text().set_text('Player is out')
        #     else:
        #         grid.get_celld()[(player_path_prev)].set_facecolor(col_map[maze[player_path_prev]])
        #         if player_path == mino_path_prev:
        #             grid.get_celld()[(player_path_prev)].get_text().set_text('Player is caught')
        #         else:
        #             grid.get_celld()[(player_path_prev)].get_text().set_text('')

        #         grid.get_celld()[(mino_path_prev)].set_facecolor(col_map[maze[mino_path_prev]])
        #         if player_path == mino_path_prev:
        #             grid.get_celld()[(mino_path_prev)].get_text().set_text('Minotaur won')
        #         else:
        #             grid.get_celld()[(mino_path_prev)].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);

    V_complete = []
    Policy_complete = []
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);
    V_complete.append(BV)
    Policy_complete.append(np.argmax(Q,1))
    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        V_complete.append(BV)
        # Show error
        #print(np.linalg.norm(V - BV))
        Policy_complete.append(np.argmax(Q,1))
    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy, V_complete, Policy_complete;


'''env = Maze(maze)
V, policy = dynamic_programming(env, 20)
print("V: ", V, "Policy: ", policy)'''