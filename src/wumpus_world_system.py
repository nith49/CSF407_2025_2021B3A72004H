import pandas as pd # type: ignore
import numpy as np  # type: ignore
import random
import math
import os
from collections import deque
from pgmpy.models import BayesianNetwork  # type: ignore
from pgmpy.factors.discrete import TabularCPD # type: ignore
from pgmpy.inference import VariableElimination # type: ignore
import matplotlib.pyplot as plt # type: ignore
from datetime import datetime

class WumpusBayesian:
    def __init__(self, size=4):
        self.arr = np.full((3, size, size), "empty", dtype=object)
        self.arr_size = size
        self.agent_pos = (0, 0)
        self.pit_num = math.floor((self.arr_size ** 2) * 0.2)
        self.wump_num = math.floor((self.arr_size ** 2) / 16)
        # Creating and store knowledge of visited cells
        self.visited = set()
        self.visit_counts = {}
        self.knowledge = np.full((size, size), "unknown", dtype=object)
        self.knowledge[0, 0] = "safe"  
        self.prob_pit_arr = np.full((size,size), 0.2)
        self.prob_wumpus_arr = np.full((size,size), self.wump_num / (self.arr_size * self.arr_size))
        self.prob_pit_arr[0][0] = 0
        self.prob_wumpus_arr[0][0] = 0
        self.move_count = 0
        
        # Create directory for saving plots
        self.plots_dir = f"wumpus_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Creating the Wumpus world
        while True:
            self.__clear_grid()
            self.__setup_pits()
            self.__setup_wump()
            self.__setup_gold()
            if self.__is_solvable():
                break
                
        # Initializing Bayesian Network
        self.setup_bayesian_network()

    def __clear_grid(self):
        self.arr.fill("empty")

    def __setup_pits(self):
        placed_pits = 0
        while placed_pits < self.pit_num:
            pit_pos = [random.randint(0, self.arr_size - 1), random.randint(0, self.arr_size - 1)]
            if pit_pos != [0, 0] and self.arr[0][pit_pos[0]][pit_pos[1]] == "empty":
                self.arr[0][pit_pos[0]][pit_pos[1]] = "pit"
                self.__add_breeze(pit_pos)
                placed_pits += 1

    def __add_breeze(self, pos):
        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.arr_size and 0 <= ny < self.arr_size and self.arr[0][nx][ny] == "empty":
                self.arr[1][nx][ny] = "breeze"

    def __setup_wump(self):
        placed_wumpus = 0
        while placed_wumpus < self.wump_num:
            wump_pos = [random.randint(0, self.arr_size - 1), random.randint(0, self.arr_size - 1)]
            if wump_pos != [0, 0] and self.arr[0][wump_pos[0]][wump_pos[1]] == "empty":
                self.arr[0][wump_pos[0]][wump_pos[1]] = "wumpus"
                self.__add_stench(wump_pos)
                placed_wumpus += 1

    def __add_stench(self, pos):
        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.arr_size and 0 <= ny < self.arr_size and self.arr[0][nx][ny] == "empty":
                self.arr[2][nx][ny] = "stench"

    def __setup_gold(self):
        while True:
            gold_pos = [random.randint(0, self.arr_size - 1), random.randint(0, self.arr_size - 1)]
            if gold_pos != [0, 0] and self.arr[0][gold_pos[0]][gold_pos[1]] == "empty":
                self.arr[0][gold_pos[0]][gold_pos[1]] = "gold"
                self.gold_pos = tuple(gold_pos)
                break

    def __is_solvable(self):
        queue = deque([(0, 0)])
        visited = set()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.popleft()
            if (x, y) == self.gold_pos:
                return True
            visited.add((x, y))
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.arr_size and 0 <= ny < self.arr_size:
                    if (nx, ny) not in visited and self.arr[0][nx][ny] != "pit" and self.arr[0][nx][ny] != "wumpus":
                        queue.append((nx, ny))
        return False
    
    def prin(self):
        print("\nWumpus World Grid:")
        print("=" * (self.arr_size * 15))
        for i in range(self.arr_size):
            for j in range(self.arr_size):
                cell_objects = []
                for layer in range(3):
                    if self.arr[layer][self.arr_size - i - 1][j] != "empty":
                        cell_objects.append(self.arr[layer][self.arr_size - i - 1][j])
                cell_str = ",".join(cell_objects) if cell_objects else "empty"
                print(f"{cell_str:^13}", end=" | ")
            print("\n" + "=" * (self.arr_size * 15))
    
    def setup_bayesian_network(self):
        """Initialize the Bayesian Network structure for Wumpus World"""
        edges = []
        nodes = []
        
        for i in range(self.arr_size):
            for j in range(self.arr_size):
                pit_node = f"P_{i}_{j}"
                nodes.append(pit_node)
                
                wumpus_node = f"W_{i}_{j}"
                nodes.append(wumpus_node)
                
                breeze_node = f"B_{i}_{j}"
                nodes.append(breeze_node)
                
                stench_node = f"S_{i}_{j}"
                nodes.append(stench_node)
                
                # Connect pit nodes to breeze nodes in adjacent cells
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < self.arr_size and 0 <= ny < self.arr_size:
                        # Pit affects breeze
                        edges.append((pit_node, f"B_{nx}_{ny}"))
                        # Wumpus affects stench
                        edges.append((wumpus_node, f"S_{nx}_{ny}"))
        
        self.model = BayesianNetwork(edges)
        self.create_cpds()
        self.inference = VariableElimination(self.model)
    
    def create_cpds(self):
        """Create Conditional Probability Distributions for all nodes"""
        cpds = []
        
        # Prior probabilities for pits and wumpuses
        pit_prior = 0.2
        wumpus_prior = self.wump_num / (self.arr_size * self.arr_size)
        
        # Create CPDs for pit and wumpus nodes (prior probabilities)
        for i in range(self.arr_size):
            for j in range(self.arr_size):
                if (i, j) == (0, 0):
                    pit_cpd = TabularCPD(
                        variable=f"P_{i}_{j}", 
                        variable_card=2, 
                        values=[[1.0], [0.0]]  
                    )
                    wumpus_cpd = TabularCPD(
                        variable=f"W_{i}_{j}", 
                        variable_card=2, 
                        values=[[1.0], [0.0]] 
                    )
                else:
                    pit_cpd = TabularCPD(
                        variable=f"P_{i}_{j}", 
                        variable_card=2, 
                        values=[[1 - pit_prior], [pit_prior]] 
                    )
                    wumpus_cpd = TabularCPD(
                        variable=f"W_{i}_{j}", 
                        variable_card=2, 
                        values=[[1 - wumpus_prior], [wumpus_prior]] 
                    )
                cpds.append(pit_cpd)
                cpds.append(wumpus_cpd)
                
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < self.arr_size and 0 <= ny < self.arr_size:
                        neighbors.append((nx, ny))
                
                # Create CPD for Breeze node based on pit neighbors
                breeze_evidence = [f"P_{nx}_{ny}" for nx, ny in neighbors]
                breeze_card = [2] * len(breeze_evidence)  # Binary variables
                
                # Calculate CPD values for breeze
                if breeze_evidence:
                    # Create truth table for all combinations of neighboring pits
                    num_combinations = 2 ** len(breeze_evidence)
                    breeze_values = []
                    
                    # For each combination, determine if breeze should be present
                    for k in range(num_combinations):
                        # Convert to binary and check if any pit is present
                        binary = format(k, f'0{len(breeze_evidence)}b')
                        # If any neighboring cell has a pit, there's a breeze
                        has_pit = '1' in binary
                        
                        if has_pit:
                            breeze_values.append([0.0, 1.0])
                        else:
                            breeze_values.append([1.0, 0.0])
                    
                    breeze_values = np.array(breeze_values).T.tolist()
                    
                    breeze_cpd = TabularCPD(
                        variable=f"B_{i}_{j}",
                        variable_card=2,
                        values=breeze_values,
                        evidence=breeze_evidence,
                        evidence_card=breeze_card
                    )
                else:
                    # No neighbors which so no possibility of breeze
                    breeze_cpd = TabularCPD(
                        variable=f"B_{i}_{j}",
                        variable_card=2,
                        values=[[1.0], [0.0]] 
                    )
                cpds.append(breeze_cpd)
                
                stench_evidence = [f"W_{nx}_{ny}" for nx, ny in neighbors]
                stench_card = [2] * len(stench_evidence)  # Binary variables
                
                if stench_evidence:
                    num_combinations = 2 ** len(stench_evidence)
                    stench_values = []
                    
                    # For each combination, determine if stench should be present
                    for k in range(num_combinations):
                        # Convert to binary and check if any wumpus is present
                        binary = format(k, f'0{len(stench_evidence)}b')
                        # If any neighboring cell has a wumpus, there's a stench
                        has_wumpus = '1' in binary
                        
                        if has_wumpus:
                            stench_values.append([0.0, 1.0])
                        else:
                            stench_values.append([1.0, 0.0])
                    
                    # Transpose for pgmpy format
                    stench_values = np.array(stench_values).T.tolist()
                    
                    stench_cpd = TabularCPD(
                        variable=f"S_{i}_{j}",
                        variable_card=2,
                        values=stench_values,
                        evidence=stench_evidence,
                        evidence_card=stench_card
                    )
                else:
                    # No neighbors, so no possibility of stench
                    stench_cpd = TabularCPD(
                        variable=f"S_{i}_{j}",
                        variable_card=2,
                        values=[[1.0], [0.0]]  # [No, Yes]
                    )
                cpds.append(stench_cpd)
        
        # Add CPDs to the model
        for cpd in cpds:
            self.model.add_cpds(cpd)
        
        # Check if the model is valid
        print(f"Model is valid: {self.model.check_model()}")
    
    def visit_cell(self, x, y):
        """Update knowledge when agent visits a cell"""
        if not (0 <= x < self.arr_size and 0 <= y < self.arr_size):
            print(f"Invalid cell coordinates: ({x}, {y})")
            return None
        
        self.visited.add((x, y))
        self.visit_counts[(x, y)] = self.visit_counts.get((x, y), 0) + 1
        self.knowledge[x, y] = "safe"
        
        evidence = {}
        
        # Check if there's a breeze
        has_breeze = self.arr[1][x][y] == "breeze"
        evidence[f"B_{x}_{y}"] = 1 if has_breeze else 0
        
        # Check if there's a stench
        has_stench = self.arr[2][x][y] == "stench"
        evidence[f"S_{x}_{y}"] = 1 if has_stench else 0
        
        self.agent_pos = (x, y)
        
        self.prob_pit_arr[x][y] = 0
        self.prob_wumpus_arr[x][y] = 0
        
        return evidence
    
    def infer_hazards(self, max_depth=None):
        """Infer hazards around the agent using the Bayesian Network"""
        if max_depth is None:
            max_depth = self.arr_size // 2 - 1
            
        if max_depth < 1:
            max_depth = 1
        
        # Get the current evidence from visited cells
        evidence = {}
        for (x, y) in self.visited:
            has_breeze = self.arr[1][x][y] == "breeze"
            evidence[f"B_{x}_{y}"] = 1 if has_breeze else 0
            
            has_stench = self.arr[2][x][y] == "stench"
            evidence[f"S_{x}_{y}"] = 1 if has_stench else 0
            
            evidence[f"P_{x}_{y}"] = 0
            evidence[f"W_{x}_{y}"] = 0
        
        x, y = self.agent_pos
        frontier = set()
        
        for i in range(max(0, x - max_depth), min(self.arr_size, x + max_depth + 1)):
            for j in range(max(0, y - max_depth), min(self.arr_size, y + max_depth + 1)):
                if (i, j) not in self.visited:
                    frontier.add((i, j))
        
        probabilities = {}
        
        for (i, j) in frontier:
            # Query for pit probability
            try:
                pit_query = self.inference.query(variables=[f"P_{i}_{j}"], evidence=evidence)
                pit_prob = pit_query.values[1]  # Probability of pit being present
                probabilities[(i, j, "pit")] = pit_prob
                self.prob_pit_arr[i][j] = pit_prob
            except Exception as e:
                print(f"Error querying pit at ({i}, {j}): {e}")
                probabilities[(i, j, "pit")] = "Error"
            
            # Query for wumpus probability
            try:
                wumpus_query = self.inference.query(variables=[f"W_{i}_{j}"], evidence=evidence)
                wumpus_prob = wumpus_query.values[1]  # Probability of wumpus being present
                probabilities[(i, j, "wumpus")] = wumpus_prob
                self.prob_wumpus_arr[i][j] = wumpus_prob
            except Exception as e:
                print(f"Error querying wumpus at ({i}, {j}): {e}")
                probabilities[(i, j, "wumpus")] = "Error"
        
        # Ensure visited cells have 0 probability
        for (i, j) in self.visited:
            self.prob_pit_arr[i][j] = 0
            self.prob_wumpus_arr[i][j] = 0
            probabilities[(i, j, "pit")] = 0
            probabilities[(i, j, "wumpus")] = 0
        
        return probabilities
    
    def visualize_probabilities(self, move_type, save=True):
        """Visualize pit and wumpus probabilities on a heatmap"""
        self.move_count += 1
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot pit probabilities
        im1 = ax1.imshow(self.prob_pit_arr, cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
        ax1.set_title(f'Pit Probabilities (Move {self.move_count} - {move_type})')
        
        for i in range(self.prob_pit_arr.shape[0]):  # Rows
            for j in range(self.prob_pit_arr.shape[1]):  # Columns
                text_color = 'white' if self.prob_pit_arr[i, j] > 0.4 else 'black'
                ax1.text(j, i, f"{self.prob_pit_arr[i, j]:.2f}", ha='center', va='center', color=text_color)
        
        ax1.plot(self.agent_pos[1], self.agent_pos[0], 'o', markersize=12, markerfacecolor='green', markeredgecolor='black')
        
        # Mark visited cells on pit plot
        for x, y in self.visited:
            if (x, y) != self.agent_pos:  # Don't overwrite the agent marker
                ax1.plot(y, x, 's', markersize=8, markerfacecolor='blue', markeredgecolor='black', alpha=0.5)
        
        # Mark gold position on pit plot with star
        gold_rect1 = plt.Rectangle((self.gold_pos[1] - 0.5, self.gold_pos[0] - 0.5), 1, 1, fill=True, color='gold', alpha=0.8)
        ax1.add_patch(gold_rect1)
        ax1.text(self.gold_pos[1], self.gold_pos[0], "Gold", ha='center', va='center', color='black', fontweight='bold')

        
        ax1.set_xticks(np.arange(self.prob_pit_arr.shape[1]))
        ax1.set_yticks(np.arange(self.prob_pit_arr.shape[0]))
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1, label="Probability")
        
        # Plot wumpus probabilities
        im2 = ax2.imshow(self.prob_wumpus_arr, cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
        ax2.set_title(f'Wumpus Probabilities (Move {self.move_count} - {move_type})')
        
        for i in range(self.prob_wumpus_arr.shape[0]):  # Rows
            for j in range(self.prob_wumpus_arr.shape[1]):  # Columns
                text_color = 'white' if self.prob_wumpus_arr[i, j] > 0.4 else 'black'
                ax2.text(j, i, f"{self.prob_wumpus_arr[i, j]:.2f}", ha='center', va='center', color=text_color)
        
        # Mark agent position on wumpus plot
        ax2.plot(self.agent_pos[1], self.agent_pos[0], 'o', markersize=12, markerfacecolor='green', markeredgecolor='black')
        
        # Mark visited cells on wumpus plot
        for x, y in self.visited:
            if (x, y) != self.agent_pos:  # Don't overwrite the agent marker
                ax2.plot(y, x, 's', markersize=8, markerfacecolor='blue', markeredgecolor='black', alpha=0.5)
        
        # Mark gold position on wumpus plot with star
        gold_rect1 = plt.Rectangle((self.gold_pos[1] - 0.5, self.gold_pos[0] - 0.5), 1, 1, fill=True, color='gold', alpha=0.8)
        ax2.add_patch(gold_rect1)
        ax2.text(self.gold_pos[1], self.gold_pos[0], "Gold", ha='center', va='center', color='black', fontweight='bold')
        
        # Customize wumpus plot axes
        ax2.set_xticks(np.arange(self.prob_wumpus_arr.shape[1]))
        ax2.set_yticks(np.arange(self.prob_wumpus_arr.shape[0]))
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        plt.colorbar(im2, ax=ax2, label="Probability")
        
        from matplotlib.lines import Line2D # type: ignore
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Agent'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Visited'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=10, label='Gold')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.1))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.plots_dir}/move_{self.move_count}_{move_type}.png", dpi=200, bbox_inches='tight')
            print(f"Saved visualization to {self.plots_dir}/move_{self.move_count}_{move_type}.png")
        else:
            plt.show()
        
        plt.close()
    
    def get_valid_moves(self):
        """Get list of valid adjacent moves from current position"""
        x, y = self.agent_pos
        valid_moves = []
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.arr_size and 0 <= ny < self.arr_size:
                direction = ""
                if dx == -1: direction = "down"
                elif dx == 1: direction = "up"
                elif dy == -1: direction = "left"
                elif dy == 1: direction = "right"
                valid_moves.append((nx, ny, direction))
        
        return valid_moves
    
    def random_move(self):
        """Make a random move to an adjacent cell"""
        print("\n--- Making a Random Move ---")
        last_pos = self.agent_pos
        valid_moves = self.get_valid_moves()
        
        if not valid_moves:
            print("No valid moves available!")
            return False
        
        # Randomly select a move
        nx, ny, direction = random.choice(valid_moves)
        print(f"Randomly moving {direction} to ({nx}, {ny})")
        
        # Check if the cell has a pit or wumpus
        if self.arr[0][nx][ny] == "pit":
            print("You fell into a pit! Restarting from last position.")
            # Restart from last position (no need to update probabilities)
            return False
        elif self.arr[0][nx][ny] == "wumpus":
            print("You were eaten by a wumpus! Restarting from last position.")
            # Restart from last position (no need to update probabilities)
            return False
        
        # Visit the new cell
        self.visit_cell(nx, ny)
        
        # Update probabilities and visualize
        self.infer_hazards()
        self.visualize_probabilities("random")
        
        # Check if gold is found
        if self.arr[0][nx][ny] == "gold":
            print("Congratulations! You found the gold with a random move!")
            return True
        
        return None  # Continue game
    
    def best_bayesian_move(self):
        """Make the best move based on Bayesian inference"""
        print("\n--- Making the Best Bayesian Move ---")
        valid_moves = self.get_valid_moves()
        
        if not valid_moves:
            print("No valid moves available!")
            return False
        
        # Update hazard probabilities
        self.infer_hazards()
        
        move_scores = []
        for nx, ny, direction in valid_moves:
              combined_risk = self.prob_pit_arr[nx][ny] + self.prob_wumpus_arr[nx][ny]
        
              revisit_penalty = self.visit_counts.get((nx, ny), 0) * 0.2
        
              exploration_bonus = 0 if (nx, ny) in self.visited else -0.1
        
              gold_distance = abs(nx - self.gold_pos[0]) + abs(ny - self.gold_pos[1])
              gold_factor = gold_distance / (2 * self.arr_size)  # Normalized distance
        
              score = combined_risk + exploration_bonus + 0.3 * gold_factor + revisit_penalty
              move_scores.append((nx, ny, direction, score))
    
        move_scores.sort(key=lambda x: x[3])
    
        nx, ny, direction, score = move_scores[0]
        print(f"Best move: {direction} to ({nx}, {ny}) (pit prob: {self.prob_pit_arr[nx][ny]:.4f})")
    

        
        if self.arr[0][nx][ny] == "pit":
            print("You fell into a pit! Game over.")
            return False
        elif self.arr[0][nx][ny] == "wumpus":
            print("You were eaten by a wumpus! Game over.")
            return False
        
        self.visit_cell(nx, ny)
        
        self.infer_hazards()
        self.visualize_probabilities("bayesian")
        
        if self.arr[0][nx][ny] == "gold":
            print("Congratulations! You found the gold with Bayesian reasoning!")
            return True
        
        return None
    
    def play_game_random(self, max_moves=50):
        """Play the game using random moves"""
        print("\nStarting Wumpus World game with Random movement strategy...")
        print("You start at position (0, 0)")
        
        self.visit_cell(0, 0)
        self.infer_hazards()
        self.visualize_probabilities("start")
        
        move_count = 0
        while move_count < max_moves:
            move_count += 1
            print(f"\nRandom Move #{move_count}")
            
            # Make a random move
            result = self.random_move()
            
            if result is True:  # Gold found
                print(f"Gold found in {move_count} random moves!")
                return True
            elif result is False:  # Pit/Wumpus encountered - continue from last position
                continue
            
            # Print current cell info
            x, y = self.agent_pos
            cell_contents = []
            if self.arr[1][x][y] == "breeze":
                cell_contents.append("breeze")
            if self.arr[2][x][y] == "stench":
                cell_contents.append("stench")
            
            if cell_contents:
                print(f"This cell contains: {', '.join(cell_contents)}")
            else:
                print("This cell is empty.")
        
        print(f"Maximum moves ({max_moves}) reached without finding gold.")
        return False
    
    def play_game_bayesian(self, max_moves=50):
        """Play the game using Bayesian inference to make the best moves"""
        print("\nStarting Wumpus World game with Bayesian reasoning strategy...")
        print("You start at position (0, 0)")
        
        # Visit the starting cell
        self.visit_cell(0, 0)
        self.infer_hazards()
        self.visualize_probabilities("start")
        
        move_count = 0
        while move_count < max_moves:
            move_count += 1
            print(f"\nBayesian Move #{move_count}")
            
            # Make the best Bayesian move
            result = self.best_bayesian_move()
            
            if result is True:  # Gold found
                print(f"Gold found in {move_count} Bayesian moves!")
                return True
            elif result is False:  # Game over
                print("Game over!")
                return False
            
            # Print current cell info
            x, y = self.agent_pos
            cell_contents = []
            if self.arr[1][x][y] == "breeze":
                cell_contents.append("breeze")
            if self.arr[2][x][y] == "stench":
                cell_contents.append("stench")
            
            if cell_contents:
                print(f"This cell contains: {', '.join(cell_contents)}")
            else:
                print("This cell is empty.")
        
        print(f"Maximum moves ({max_moves}) reached without finding gold.")
        return False

def compare_strategies(grid_size=4, num_trials=10, max_moves=50):
    """Compare random vs. Bayesian strategies across multiple trials"""
    results = {
        "random": {"wins": 0, "avg_moves": []},
        "bayesian": {"wins": 0, "avg_moves": []}
    }
    
    for trial in range(1, num_trials + 1):
        print(f"\n==== TRIAL {trial}/{num_trials} ====")
        
        # Create a new game instance for random strategy
        print("\n=== Testing Random Strategy ===")
        random_game = WumpusBayesian(size=grid_size)
        random_game.prin()
        random_start_time = datetime.now()
        random_success = random_game.play_game_random(max_moves)
        random_end_time = datetime.now()
        random_time = (random_end_time - random_start_time).total_seconds()
        
        if random_success:
            results["random"]["wins"] += 1
            results["random"]["avg_moves"].append(random_game.move_count)
        
        # Create a new game instance with same configuration for Bayesian strategy
        print("\n=== Testing Bayesian Strategy ===")
        bayesian_game = WumpusBayesian(size=grid_size)
        bayesian_game.prin()
        bayesian_start_time = datetime.now()
        bayesian_success = bayesian_game.play_game_bayesian(max_moves)
        bayesian_end_time = datetime.now()
        bayesian_time = (bayesian_end_time - bayesian_start_time).total_seconds()
        
        if bayesian_success:
            results["bayesian"]["wins"] += 1
            results["bayesian"]["avg_moves"].append(bayesian_game.move_count)
        
        print(f"\n--- Trial {trial} Results ---")
        print(f"Random Strategy: {'Success' if random_success else 'Failure'} in {random_game.move_count} moves ({random_time:.2f} seconds)")
        print(f"Bayesian Strategy: {'Success' if bayesian_success else 'Failure'} in {bayesian_game.move_count} moves ({bayesian_time:.2f} seconds)")
    
    # Overall results
    print("\n==== OVERALL RESULTS ====")
    print(f"Random Strategy: {results['random']['wins']}/{num_trials} successes ({results['random']['wins']/num_trials*100:.1f}%)")
    if results['random']['avg_moves']:
        print(f"  Average moves to win: {sum(results['random']['avg_moves'])/len(results['random']['avg_moves']):.2f}")
    
    print(f"Bayesian Strategy: {results['bayesian']['wins']}/{num_trials} successes ({results['bayesian']['wins']/num_trials*100:.1f}%)")
    if results['bayesian']['avg_moves']:
        print(f"  Average moves to win: {sum(results['bayesian']['avg_moves'])/len(results['bayesian']['avg_moves']):.2f}")
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['Random', 'Bayesian'], 
            [results['random']['wins']/num_trials*100, results['bayesian']['wins']/num_trials*100],
            color=['blue', 'orange'])
    plt.title('Win Rate Comparison')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for i, v in enumerate([results['random']['wins']/num_trials*100, results['bayesian']['wins']/num_trials*100]):
        plt.text(i, v + 3, f"{v:.1f}%", ha='center')
    
    plt.savefig("strategy_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # If there are wins then compare average moves
    if results['random']['avg_moves'] and results['bayesian']['avg_moves']:
        plt.figure(figsize=(10, 6))
        plt.bar(['Random', 'Bayesian'], 
                [sum(results['random']['avg_moves'])/len(results['random']['avg_moves']), 
                 sum(results['bayesian']['avg_moves'])/len(results['bayesian']['avg_moves'])],
                color=['blue', 'orange'])
        plt.title('Average Moves to Win Comparison')
        plt.ylabel('Average Moves')
        
        for i, v in enumerate([sum(results['random']['avg_moves'])/len(results['random']['avg_moves']), 
                               sum(results['bayesian']['avg_moves'])/len(results['bayesian']['avg_moves'])]):
            plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
        
        plt.savefig("moves_comparison.png", dpi=200, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("Wumpus World with Bayesian Reasoning")
    print("====================================")
    
    grid_size = int(input("Enter the Grid Size (N >= 4): ")) 
    num_trials = 5  
    max_moves = int(input("Enter the maximum number of moves: "))  
    
    # Compare strategies
    compare_strategies(grid_size, num_trials, max_moves)
    
    print("\nDone! Please Check the generated plots for visualization of the results.")