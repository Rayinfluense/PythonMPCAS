from collections import deque

import numpy as np
import random
import math
import h5py
import copy
import matplotlib.pyplot as plt
from qNet import QNet
import torch.optim as optim
import torch.nn as nn
import torch

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard

        rows = gameboard.N_row
        columns = gameboard.N_col
        n_tiles = len(gameboard.tiles)
        episode_count = self.episode_count

        self.interval_rewards = []
        self.interval_q_table = []
        self.q_table = np.zeros([2**(rows*columns)*n_tiles,n_tiles*4], dtype=np.float32) #Binary for tiles and multiplier for place piece number.
        self.q_table_hist = []
        self.q_state_index = 0 #Index in q_table for the state we're currently in
        self.reward = 0 #Reward for last move
        self.reward_tots = np.zeros([episode_count]) #List of full-game rewards for each game played
        self.old_state_index = 0 #Index for previous state.

        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions and Q-table and storage for the rewards
        # This function should not return a value, store Q table etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.episode_count' the total number of episodes in the training

    def fn_load_strategy(self,strategy_file):
        strategy_table = np.genfromtxt(strategy_file, delimiter=',')
        self.q_table = strategy_table
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)

    def fn_read_state(self):
        board_mat = self.gameboard.board
        n_col = self.gameboard.N_col
        n_row = self.gameboard.N_row
        binary_list = np.zeros(n_col*n_row)
        for i in range(n_col):
            for j in range(n_row):
                if board_mat[i,j] == 1:
                    binary_list[i+4*j] = 1

        self.q_state_index = 0
        for i in range(len(binary_list)):
            self.q_state_index += binary_list[i] * (2 ** (i))

        self.q_state_index += (2 ** (n_row * n_col)) * (self.gameboard.cur_tile_type)
        self.q_state_index = round(self.q_state_index)
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as an integer entry in the Q-table
        # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        select_move = True
        illegal_moves_indices = []
        while select_move:
            if random.random() < self.epsilon:
                self.action_index = random.randint(0, self.gameboard.N_col*4 - 1)
            else:
                #q_state_vec = copy.deepcopy(self.q_table[self.q_state_index, :])
                q_state_vec = self.q_table[self.q_state_index,:]

                for i in illegal_moves_indices:
                    q_state_vec[i] = -float('inf') #Remove illegal moves

                max_value = -float('inf') #Less than instant loss
                candidates = []
                action_index = 0
                for elem in q_state_vec:
                    if elem > max_value:
                        max_value = elem
                        candidates = [action_index]
                    elif elem == max_value:
                        max_value = elem
                        candidates.append(action_index)
                    action_index += 1
                #print(q_state_vec)
                self.action_index = random.choice(candidates)

            tile_x = math.floor(self.action_index / self.gameboard.N_col)
            tile_orientation = self.action_index % 4
            select_move_int = self.gameboard.fn_move(tile_x,tile_orientation)
            if select_move_int == 0:
                select_move = False
            else:
                illegal_moves_indices.append(self.action_index)
                select_move = True

        # Choose and execute an action, based on the Q-table or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not
    
    def fn_reinforce(self,old_state,reward):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q table using state and action stored as attributes in self and using function arguments for the old state and the reward
        # This function should not return a value, the Q table is stored as an attribute of self
        alpha = self.alpha
        q_index = self.q_state_index
        self.q_table[old_state,self.action_index] += alpha*(reward + np.max(self.q_table[q_index,:]) - self.q_table[old_state,self.action_index])
        # Useful variables: 
        # 'self.alpha' learning rate

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                self.interval_rewards.append(np.sum(self.reward_tots[range(self.episode-100,self.episode)])/100)
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    self.interval_q_table.append(self.q_table)
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
            if self.episode>=self.episode_count:
                strategy_file = self.q_table
                np.savetxt("strategy_file.csv", strategy_file, delimiter=",")
                x = list(range(0,self.episode_count,100))
                plt.plot(x,self.interval_rewards)
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.show()
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()
            old_state = self.q_state_index
            # Drop the tile on the game board
            self.reward=self.gameboard.fn_drop()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += self.reward
            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state,self.reward)


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count

    class ReplayMemory:

        def __init__(self, capacity):
            self.memory = deque([], maxlen = capacity)

        def push(self, replay): #Replay will be tuple (quadruple) of (state, action, reward, new state)
            self.memory.append(replay)

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self
        hidden_layer_width = 64

        self.interval_rewards = []
        self.episode_number = 0
        #The net will want a vector as input.
        self.old_board_state = torch.from_numpy(np.zeros([self.gameboard.N_col * self.gameboard.N_row + 1], dtype=np.float32))
        self.board_state = torch.from_numpy(np.zeros([self.gameboard.N_col * self.gameboard.N_row + 1], dtype=np.float32)) #List of all tile position states.
        self.action_index = 0

        self.net = QNet(self.gameboard, hidden_layer_width)
        self.netHat = QNet(self.gameboard, hidden_layer_width)

        self.experience_replay = TDQNAgent.ReplayMemory(self.replay_buffer_size) #Use self.experience_replay.push(replay) to push a quadruplet of replay.

        self.reward = 0 #Reward for last move
        self.reward_tots = np.zeros([self.episode_count]) #List of full-game rewards for each game played
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.alpha) #Using the Adam optimizer
        self.criterion = nn.MSELoss()

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self,strategy_file):
        #CHANGE TO NEURAL NETWORK EQUILVALENT
        strategy_table = np.genfromtxt(strategy_file, delimiter=',')
        self.net = strategy_table
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self
        self.old_board_state = copy.deepcopy(self.board_state)
        flattened_state = np.matrix.flatten(self.gameboard.board)

        self.board_state = torch.from_numpy(np.append(flattened_state, self.gameboard.cur_tile_type))  #Add the tile type at the end to extend the state
        self.board_state = self.board_state.to(torch.float32)
        # Useful variables:
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        select_move = True
        illegal_moves_indices = []
        while select_move:
            if random.random() < max(self.epsilon, 1 - self.episode/self.epsilon_scale):
                self.action_index = random.randint(0, self.gameboard.N_col * 4 - 1)
            else:
                q_state_vec = self.net.forward(self.board_state) #Outputs of neural network
                q_state_vec = q_state_vec.detach().cpu().numpy()
                print("Ouptut was" ,q_state_vec)

                for i in illegal_moves_indices:
                    q_state_vec[i] = -float('inf')  #Doesn't permanently fix illegal moves :/

                max_value = -float('inf')  # Less than instant loss
                candidates = []
                action_index = 0
                for elem in q_state_vec:
                    if elem > max_value:
                        max_value = elem
                        candidates = [action_index]
                    elif elem == max_value:
                        max_value = elem
                        candidates.append(action_index)
                    action_index += 1
                #print(q_state_vec)
                self.action_index = random.choice(candidates)

            tile_x = math.floor(self.action_index / self.gameboard.N_col)
            tile_orientation = self.action_index % 4
            select_move_int = self.gameboard.fn_move(tile_x, tile_orientation)
            if select_move_int == 0:
                select_move = False
            else:
                illegal_moves_indices.append(self.action_index) #Doesn't permanently fix illegal moves :/
                select_move = True


        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)

        #Calculate loss according to replay and adjust weights according to Adam scheme.
        samples = self.experience_replay.sample(self.batch_size)

        for replay in samples:
            #Perform backprop and adjustment
            self.optimizer.zero_grad()
            old_state = replay[0] #Comes as tensor which is fine
            action = replay[1] #Number
            reward = replay[2] #Always number
            new_state = replay[3]
            output = self.net(old_state)[action] #Comes as tensor
            print("Selected output ", output)
            y = reward + max(self.netHat(new_state))
            print("r = ", reward)
            print("y = ", y)
            loss = self.criterion(output, y)
            print("loss is ", loss)
            loss.backward()
            self.optimizer.step()

        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                #for param in self.net.parameters():
                #    print(param.data.detach().numpy())
                #print(list(self.net.parameters()))
                self.interval_rewards.append(np.sum(self.reward_tots[range(self.episode-100,self.episode)])/100)
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                strategy_file = self.net.state_dict()
                torch.save(strategy_file, "strategy_file.pt")
                x = list(range(0, self.episode_count, 100))
                plt.plot(x, self.interval_rewards)
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.show()
                raise SystemExit(0)
            else:
                if (len(self.experience_replay) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    self.netHat = copy.deepcopy(self.net)
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

            # Drop the tile on the game board
            self.reward=self.gameboard.fn_drop()
            self.reward_tots[self.episode] += self.reward
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            # Read the new state
            self.fn_read_state()
            replay = (self.old_board_state, self.action_index, self.reward, self.board_state)
            self.experience_replay.push(replay)
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.experience_replay) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                self.fn_reinforce()



class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()