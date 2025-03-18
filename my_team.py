# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(
    first_index,
    second_index,
    is_red,
    first="SingleSmartAgent",
    second="SingleSmartAgent",
    num_training=0,
):
    """
    For both positions, we use SingleSmartAgent. We rely on in-game distance checks
    to decide which agent defends or attacks at runtime.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


class AStarPlanner:
    def __init__(self, walls):
        self.walls = walls

    def get_neighbors(self, position):
        x, y = int(position[0]), int(position[1])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.walls.width
                and 0 <= ny < self.walls.height
                and not self.walls[nx][ny]
            ):
                neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def search(self, start, goal):
        frontier = util.PriorityQueue()
        frontier.push(start, 0)
        cost_so_far = {start: 0}

        while not frontier.is_empty():
            current = frontier.pop()
            if current == goal:
                return cost_so_far[current]

            for neighbor in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    frontier.push(neighbor, priority)
        return 999999  # Large cost if goal not reached

    def safe_search(self, start, goal, ghost_positions):
        # not none positions
        ghost_positions = [g for g in ghost_positions if g is not None]

        frontier = util.PriorityQueue()
        frontier.push(start, 0)
        cost_so_far = {start: 0}

        while not frontier.is_empty():
            current = frontier.pop()
            if current == goal:
                return cost_so_far[current]

            for neighbor in self.get_neighbors(current):
                if neighbor in ghost_positions or any(
                    self.heuristic(neighbor, gp) <= 1 for gp in ghost_positions
                ):
                    new_cost = 999999  # Large cost if ghost is in the way
                    continue  # Skip neighbors that are ghost positions
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    frontier.push(neighbor, priority)
        return 999999  # Large cost if goal not reached


class SingleSmartAgent(CaptureAgent):
    """
    A single agent class that can switch between offensive and defensive roles
    based on distance to its spawn position, plus an additional check to remain
    offensive if Pac-Man is powered with time > 10.
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.role = "attack"  # Default role
        self.random_frontier = None
        self.closest_safe_frontier = None
        self.powered = False

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.static_walls = game_state.get_walls()  # Cache static maze layout
        self.walkable_frontier_positions = self.get_frontier_positions(
            self.static_walls
        )
        self.random_frontier = random.choice(self.walkable_frontier_positions)
        CaptureAgent.register_initial_state(self, game_state)

    def get_frontier_positions(self, walls):
        layout_width, layout_height = walls.width, walls.height

        # If red, use left half; if blue, use right half
        if self.red:
            frontier_col = layout_width // 2 - 1
        else:
            frontier_col = layout_width // 2

        walkable_frontier_positions = []
        for row in range(layout_height):
            if not walls[frontier_col][row]:
                walkable_frontier_positions.append((frontier_col, row))

        return walkable_frontier_positions

    # get random valid frontier position (for defender)
    def get_random_frontier_position(self, prob=0.1):
        if self.walkable_frontier_positions:
            chosen_pos = self.random_frontier
            if random.random() < prob and len(self.walkable_frontier_positions) > 1:
                alt_positions = [
                    p for p in self.walkable_frontier_positions if p != chosen_pos
                ]
                chosen_pos = random.choice(alt_positions)
            self.random_frontier = chosen_pos

    # get closest valid frontier position (for safe return)
    def get_closest_frontier_position(self, game_state):
        my_pos = game_state.get_agent_position(self.index)

        if self.walkable_frontier_positions:
            distances = [
                self.get_maze_distance(my_pos, p)
                for p in self.walkable_frontier_positions
            ]
            closest_pos = self.walkable_frontier_positions[
                distances.index(min(distances))
            ]
            self.closest_safe_frontier = closest_pos

    def choose_action(self, game_state):
        # Update role without explicit override conditions
        self.update_role(game_state)

        self.get_random_frontier_position()
        self.get_closest_frontier_position(game_state)

        # Get legal actions and current position
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_position(self.index)
        walls = game_state.get_walls()
        layout_width = walls.width

        # Determine if we are on enemy side (if red: right half; else: left half)
        on_enemy_side = (
            (my_pos[0] >= layout_width // 2)
            if self.red
            else (my_pos[0] < layout_width // 2)
        )

        # Remove STOP action when on enemy side.
        if on_enemy_side and Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # Epsilon-greedy random action selection.
        epsilon = 0.05
        if random.random() < epsilon:
            return random.choice(actions)

        # Evaluate actions based on features & weights.
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        best_values = [v for a, v in zip(actions, values) if v == max_value]

        # if self.role == "attack":
        #     print(f"Agent {self.index} has {best_values} for {best_actions}")

        choosen_action = random.choice(best_actions)

        return choosen_action

    def update_role(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        my_dist = self.a_star_distance(game_state, my_pos, self.start, dynamic=False)
        teammates = self.get_team(game_state)
        other_idx = [i for i in teammates if i != self.index]
        if not other_idx:
            return  # Single-agent corner case

        other_pos = game_state.get_agent_position(other_idx[0])
        other_dist = self.a_star_distance(
            game_state, other_pos, self.start, dynamic=False
        )
        is_closer = my_dist < other_dist

        # Check power state: if I'm Pac-Man and "powered" timer > 10, enforce attack
        enemies = [
            game_state.get_agent_state(i) for i in self.get_opponents(game_state)
        ]
        self.powered = False
        for e in enemies:
            # If an enemy is a ghost and scared, check if timer > 5
            if not e.is_pacman and e.scared_timer and e.scared_timer > 5:
                self.powered = True
                break

        # Check if there's an invader on our side
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        # intruder_nearby = False
        # if not game_state.get_agent_state(self.index).is_pacman and invaders:
        #     if self.powered:
        #         intruder_nearby = False
        #     else:
        #         valid_invaders = [
        #             inv
        #             for inv in invaders
        #             if inv.scared_timer is None or inv.scared_timer < 5
        #         ]
        #         if valid_invaders:
        #             intruder_nearby = True
        #             distances = [
        #                 self.get_maze_distance(my_pos, inv.get_position())
        #                 for inv in valid_invaders
        #             ]
        #             if distances and min(distances) < 5:
        #                 intruder_nearby = True

        # if i am scared, attack
        if game_state.get_agent_state(self.index).scared_timer:
            self.role = "attack"
            # return

        distance_between_teammates = self.get_maze_distance(my_pos, other_pos)

        if self.powered:
            self.role = "attack"
        # intruder_nearby and not teammate is closer
        elif invaders and distance_between_teammates > 5:
            self.role = "defend"
        else:
            self.role = "defend" if is_closer else "attack"

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        # Merge "offensive" and "defensive" features/weights
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    ########### OFFENSIVE HELPERS ###########

    def a_star_distance(self, game_state, start, goal, dynamic=True, safe=False):
        if dynamic:
            walls = game_state.get_walls()
        else:
            walls = self.static_walls
        planner = AStarPlanner(walls)

        if safe:
            enemies = [
                game_state.get_agent_state(i) for i in self.get_opponents(game_state)
            ]
            ghost_positions = [e.get_position() for e in enemies if not e.is_pacman]
            return planner.safe_search(start, goal, ghost_positions)
        else:
            return planner.search(start, goal)

    def get_neighbors(self, position, walls):
        x, y = int(position[0]), int(position[1])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = int(x + dx), int(y + dy)
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                neighbors.append((nx, ny))
        return neighbors

    ########### FEATURE EXTRACTION ###########

    def get_features(self, game_state, action):
        successor = self.get_successor(game_state, action)
        features = util.Counter()

        # Determine if we're on our side or enemy side
        walls = game_state.get_walls()
        layout_width = walls.width
        my_pos = successor.get_agent_state(self.index).get_position()
        my_x = my_pos[0]
        on_own_side = (
            (my_x < layout_width // 2) if self.red else (my_x >= layout_width // 2)
        )

        # If I'm on enemy side and an enemy is close, penalty
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        enemy_positions = [
            e.get_position() for e in enemies if e.get_position() is not None
        ]
        if enemy_positions:
            dists = [self.get_maze_distance(my_pos, ep) for ep in enemy_positions]
            closest_enemy = min(dists)
            if not on_own_side and closest_enemy < 3:
                features["distance_to_closest_enemy"] = closest_enemy

        # Use role-based features. Note the modified call for offensive features:
        if self.role == "attack":
            features.update(self.get_offensive_features(game_state, successor))
        else:
            features.update(self.get_defensive_features(successor))

        # if self.role == "attack":
        #     print("Action", action)

        return features

    def get_offensive_features(self, game_state, successor):
        features = util.Counter()
        my_pos = successor.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        layout_width = walls.width

        # Base features.
        food_list = self.get_food(successor).as_list()
        features["successor_score"] = -len(food_list)
        if food_list:
            features["distance_to_food"] = min(
                self.a_star_distance(successor, my_pos, food, dynamic=False, safe=True)
                for food in food_list
            )

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [
            a for a in enemies if not a.is_pacman and a.get_position() is not None
        ]
        if ghosts:
            ghost_distances = [
                self.get_maze_distance(my_pos, g.get_position()) for g in ghosts
            ]
            closest_ghost = min(ghost_distances)
            # if ghost is very close while on enemy side, we are being chased.
            on_enemy_side = (
                (my_pos[0] >= layout_width // 2)
                if self.red
                else (my_pos[0] < layout_width // 2)
            )
            if on_enemy_side and closest_ghost < 4 and not self.powered:
                features["being_chased"] = 1
                features["distance_to_food"] == 9999
                capsules = self.get_capsules(successor)
                if capsules:
                    features["capsule_distance"] = min(
                        self.a_star_distance(successor, my_pos, cap, dynamic=False)
                        for cap in capsules
                    )
                else:
                    features["fleeing"] = 5 - closest_ghost
            elif on_enemy_side and closest_ghost < 4 and self.powered:
                # hunt the ghost
                features["distance_to_food"] = closest_ghost
            else:
                features["being_chased"] = 0

        # More dynamic return-home logic
        carrying = successor.get_agent_state(self.index).num_carrying
        if carrying > 0:
            # Calculate risk vs reward
            distance_home = self.a_star_distance(
                successor,
                my_pos,
                self.closest_safe_frontier,
                dynamic=False,
                safe=True,
            )
            time_remaining = game_state.data.timeleft

            # More urgent return if:
            # 1. Carrying lots of food
            # 2. Almost out of time
            # 3. Significant score advantage to protect
            score = self.get_score(game_state)

            # Adjust threshold based on game state
            return_threshold = 5
            if time_remaining < 200:  # Late game
                return_threshold = 3
            elif score > 5:  # Protecting a lead
                return_threshold = 4

            if carrying >= return_threshold or time_remaining < (distance_home + 6) * 4:
                features["return_home"] = distance_home * (1 + carrying / 5)

        # if self.role == "attack":
        #     print(f"Agent {self.index} has {features}")

        return features

    def get_defensive_features(self, successor):
        features = util.Counter()
        my_pos = successor.get_agent_state(self.index).get_position()

        # Existing defensive features
        features["on_defense"] = (
            1 if not successor.get_agent_state(self.index).is_pacman else 0
        )
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features["num_invaders"] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            min_dist = min(dists)
            # if i am scared, run away from the closest invader
            if successor.get_agent_state(self.index).scared_timer:
                features["invader_distance"] = -min_dist
            else:
                features["invader_distance"] = min_dist

        if self.random_frontier:
            features["distance_to_frontier"] = self.a_star_distance(
                successor, my_pos, self.random_frontier, dynamic=False, safe=True
            )

        return features

    ########### WEIGHTING ###########

    def get_weights(self, game_state, action):
        if self.role == "attack":
            return {
                "successor_score": 100,
                "distance_to_food": -1,
                "return_home": -2,
                "capsule_distance": -3,
                "fleeing": -10,
                "distance_to_closest_enemy": 5,
                "being_chased": -2000,  # Strong penalty if being chased by a ghost in enemy territory
            }
        else:
            return {
                "num_invaders": -1000,
                "on_defense": 100,
                "invader_distance": -10,
                "distance_to_frontier": -1,
                "distance_to_closest_enemy": 5,
            }
