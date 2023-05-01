# import random

# class Node:
# 	def __init__(self, state, parent=None):
# 		self.state = state
# 		self.parent = parent
# 		self.children = []
# 		self.wins = 0
# 		self.visits = 0

# 	def add_child(self, child_state):
# 		child = Node(child_state, self)
# 		self.children.append(child)
# 		return child

# 	def select_child(self):
# 		return max(self.children, key=lambda node: node.ucb())

# 	def ucb(self, c=1.4):
# 		if self.visits == 0:
# 			return float('inf')
# 		return self.wins / self.visits + c * (math.log(self.parent.visits) / self.visits) ** 0.5

# 	def update(self, result):
# 		self.visits += 1
# 		self.wins += result

# 	def is_terminal(self):
# 		return self.state.is_terminal()

# 	def is_fully_expanded(self):
# 		return len(self.children) == len(self.state.get_possible_moves())

# 	def simulate(self):
# 		state = self.state.clone()
# 		while not state.is_terminal():
# 			move = random.choice(state.get_possible_moves())
# 			state.apply_move(move)
# 		return state.get_result()

# 	def rollout(self):
# 		node = self
# 		while not node.is_terminal():
# 			if not node.is_fully_expanded():
# 				return node.expand()
# 			node = node.select_child()
# 		result = node.simulate()
# 		node.backpropagate(result)
# 		return result

# 	def expand(self):
# 		move = random.choice(self.state.get_possible_moves())
# 		child_state = self.state.clone()
# 		child_state.apply_move(move)
# 		return self.add_child(child_state)

# 	def backpropagate(self, result):
# 		self.update(result)
# 		if self.parent:
# 			self.parent.backpropagate(result)

# class State:
# 	def __init__(self, board, position):
# 		self.board = board
# 		self.position = position

# 	def __eq__(self, other):
# 		return self.board == other.board and self.position == other.position

# 	def clone(self):
# 		return State([row[:] for row in self.board], self.position)

# 	def apply_move(self, move):
# 		row, col = self.position
# 		if move == 'left':
# 			if col == 0 or self.board[row][col-1] == 'X':
# 				raise ValueError('Invalid move')
# 			if len(self.board[row]) > col+1 and self.board[row][col+1] == 'X':
# 				self.board[row][col+1] = 'O'
# 			self.position = (row, col-1)
# 		elif move == 'right':
# 			if col == len(self.board[row])-1 or self.board[row][col+1] == 'X':
# 				raise ValueError('Invalid move')
# 			if col > 0 and self.board[row][col-1] == 'X':
# 				self.board[row][col-1] = 'O'
# 			self.position = (row, col+1)
# 		elif move == 'up':
# 			if row == 0 or self.board[row-1][col] == 'X':
# 				raise ValueError('Invalid move')
# 			if len(self.board) > row+1 and self.board[row+1][col] == 'X':
# 				self.board[row+1][col] = 'O'
# 			self.position = (row-1, col)
# 		elif move == 'down':
# 			if row == len(self.board)-1 or self.board[row+1][col] == 'X':
# 				raise ValueError('Invalid move')
# 			if row > 0 and self.board[row-1][col] == 'X':
# 				self.board[row-1][col] = 'O'
# 			self.position = (row+1, col)
# 		else:
# 			raise ValueError('Invalid move')

# 	def get_possible_moves(self):
# 		moves = []
# 		row, col = self.position
# 		if col > 0 and self.board[row][col-1] != 'X':
# 			moves.append('left')
# 		if col < len(self.board[row])-1 and self.board[row][col+1] != 'X':
# 			moves.append('right')
# 		if row > 0 and self.board[row-1][col] != 'X':
# 			moves.append('up')
# 		if row < len(self.board)-1 and self.board[row+1][col] != 'X':
# 			moves.append('down')
# 		return moves

# 	def is_terminal(self):
# 		row, col = self.position
# 		if self.board[row][col] == 'E':
# 			return True
# 		if len(self.get_possible_moves()) == 0:
# 			return True
# 		return False

# 	def get_result(self):
# 		row, col = self.position
# 		if self.board[row][col] == 'E':
# 			return 1.0
# 		return 0.0

# 	def __str__(self):
# 		return '\n'.join([''.join(row) for row in self.board])

# def simulate(state):
# 	node = Node(state)
# 	for i in range(1000):
# 		node.rollout()
# 	return node.select_child().state

# board = [
# 	['O', 'O', 'O', 'O'],
# 	['X', 'X', 'X', 'X'],
# 	['O', 'O', 'O', 'O'],
# 	['O', 'E', 'O', 'O'],
# 	['O', 'X', 'X', 'O']
# ]
# state = State(board, (3, 1))
# resulting_state = simulate(state)
# print(resulting_state)


import numpy as np
import random

"""
Creating Block class to represent blocks in the game
Class Variables:
self.width (integer) -- width of block
self.length (integer) -- length of block
self.blocks (numpy array of integers) -- 2D numpy array to represent block configuration
    0 -- empty space
    1 -- block that needs to be moved
    2 -- block that serves as support
    3 -- checkpoint
self.start (tuple of integers) -- starting position of the block
self.goal (tuple of integers) -- goal position of block
self.current (tuple of integers) -- current position of block

Methods:
rotate() -- rotates the block by 90 degrees
is_valid_move(direction) -- returns True if moving in given direction is valid, else False
move(direction) -- moves the block in given direction
reset() -- resets the block to the starting position
"""

import random
import copy

# Define the various directions for movement
UP = 'u'
DOWN = 'd'
LEFT = 'l'
RIGHT = 'r'

class Block:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Bloxorz:
    def __init__(self, grid, target):
        self.grid = grid
        self.target = target
        self.width = len(grid[0])
        self.height = len(grid)
        self.block = self.find_block()
        self.moves = []

    def find_block(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 'B':
                    return Block(x, y)

    def get_block_position(self):
        if self.block.x == self.target[0] and self.block.y == self.target[1]:
            return "win"
        elif self.grid[self.block.y][self.block.x] == 'X':
            return "lose"
        elif self.grid[self.block.y][self.block.x] in ['B', 'S']:
            return "ok"
        else:
            raise Exception("Invalid block position")

    def get_valid_moves(self):
        moves = []
        if self.block.x == self.target[0] and self.block.y == self.target[1]:
            return moves
        if self.block.x == self.target[0]:  # vertical position
            if self.block.y > 0 and self.grid[self.block.y - 1][self.block.x] != 'X':
                moves.append(UP)
            if self.block.y < self.height - 1 and self.grid[self.block.y + 1][self.block.x] != 'X':
                moves.append(DOWN)
        else:  # horizontal position
            if self.block.x > 0 and self.grid[self.block.y][self.block.x - 1] != 'X':
                moves.append(LEFT)
            if self.block.x < self.width - 1 and self.grid[self.block.y][self.block.x + 1] != 'X':
                moves.append(RIGHT)
        return moves

    def move_block(self, move):
        new_block = copy.deepcopy(self.block)
        if move == LEFT:
            new_block.x -= 1
        elif move == RIGHT:
            new_block.x += 1
        elif move == UP:
            new_block.y -= 1
        elif move == DOWN:
            new_block.y += 1
        self.block = new_block

    def make_move(self, move):
        self.move_block(move)
        result = self.get_block_position()
        if result == "ok":
            self.moves.append(move)
        return result

    def undo_move(self):
        if len(self.moves) == 0:
            return
        move = self.moves.pop()
        opposite_move = ''
        if move == UP:
            opposite_move = DOWN
        elif move == DOWN:
            opposite_move = UP
        elif move == LEFT:
            opposite_move = RIGHT
        elif move == RIGHT:
            opposite_move = LEFT
        self.make_move(opposite_move)

class MCTSNode:
    def __init__(self, state, move = None, parent = None):
        self.visit_count = 0
        self.reward = 0
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.untried_moves = state.get_valid_moves()

    def select_child(self):
        return max(self.children, key=lambda node: node.get_uct())

    def get_uct(self):
        exploitation_term = self.reward / (self.visit_count + 1)
        exploration_term = math.sqrt(math.log(self.parent.visit_count) / (self.visit_count + 1))
        return exploitation_term + EXPLORATION_PARAM * exploration_term

    def add_child(self, child_state, move):
        child_node = MCTSNode(child_state, move, self)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.reward += reward
        self.visit_count += 1

class MCTSSolver:
    def __init__(self, grid, target):
        self.initial_state = Bloxorz(grid, target)

    def solve(self):
        root_node = MCTSNode(self.initial_state)

        for i in range(NUM_SIMULATIONS):
            node = root_node
            state = copy.deepcopy(self.initial_state)

            # Selection
            while len(node.untried_moves) == 0 and len(node.children) > 0:
                node = node.select_child()
                state.make_move(node.move)

            # Expansion
            if len(node.untried_moves) > 0:
                random_move = random.choice(node.untried_moves)
                state.make_move(random_move)
                node = node.add_child(state, random_move)

            # Simulation
            while True:
                valid_moves = state.get_valid_moves()
                if len(valid_moves) == 0:
                    break
                random_move = random.choice(valid_moves)
                state.make_move(random_move)
                result = state.get_block_position()
                if result != 'ok':
                    break

            # Backpropagation
            reward = 0
            if result == 'win':
                reward = 1
            elif result == 'lose':
                reward = -1

            while node is not None:
                node.update(reward)
                node = node.parent

        # Choose the best move
        best_child = root_node.select_child()
        return best_child.move



