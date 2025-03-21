from base import Agent, Action, Perception
from representation import GridRelativeOrientation, GridOrientation
from communication import SocialAction, AgentMessage
from hunting import HuntingEnvironment, WildLifeAgentData, WildLifeAgent
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt

import time, random

class ProbabilityMap(object):

    def __init__(self, existing_map = None):
        self.__internal_dict = {}

        if existing_map:
            for k, v in existing_map.list_actions():
                self.__internal_dict[k] = v

    def empty(self):
        if self.__internal_dict:
            return False

        return True


    def put(self, action, value):
        self.__internal_dict[action] = value

    def remove(self, action):
        """
        Updates a discrete action probability map by uniformly redistributing the probability of an action to remove over
        the remaining possible actions in the map.
        :param action: The action to remove from the map
        :return:
        """
        if action in self.__internal_dict:
            val = self.__internal_dict[action]
            del self.__internal_dict[action]

            remaining_actions = list(self.__internal_dict.keys())
            nr_remaining_actions = len(remaining_actions)

            if nr_remaining_actions != 0:
                prob_sum = 0
                for i in range(nr_remaining_actions - 1):
                    new_action_prob = (self.__internal_dict[remaining_actions[i]] + val) / float(nr_remaining_actions)
                    prob_sum += new_action_prob

                    self.__internal_dict[remaining_actions[i]] = new_action_prob

                self.__internal_dict[remaining_actions[nr_remaining_actions - 1]] = 1 - prob_sum


    def choice(self):
        """
        Return a random action from a discrete distribution over a set of possible actions.
        :return: an action chosen from the set of choices
        """
        r = random.random()
        count_prob = 0

        for a in self.__internal_dict.keys():
            count_prob += self.__internal_dict[a]
            if count_prob >= r:
                return a

        raise RuntimeError("Should never get to this point when selecting an action")

    def list_actions(self):
        return self.__internal_dict.items()


class MyAction(Action, Enum):
    """
    Physical actions for wildlife agents.
    """

    # The agent must move north (up)
    NORTH = 0

    # The agent must move east (right).
    EAST = 1

    # The agent must move south (down).
    SOUTH = 2

    # The agent must move west (left).
    WEST = 3



class MyAgentPerception(Perception):
    """
    The perceptions of a wildlife agent.
    """

    def __init__(self, agent_position, obstacles, nearby_predators, nearby_prey, messages = None):
        """
        Default constructor
        :param agent_position: agents's position.
        :param obstacles: visible obstacles
        :param nearby_predators: visible predators - given as tuple (agent_id, grid position)
        :param nearby_prey: visible prey - given as tuple (agent_id, grid_position)
        :param messages: incoming messages, may be None
        """
        self.agent_position = agent_position
        self.obstacles = obstacles
        self.nearby_predators = nearby_predators
        self.nearby_prey = nearby_prey

        if messages:
            self.messages = messages
        else:
            self.messages = []


class MyPrey(WildLifeAgent):
    """
    Implementation of the prey agent.
    """
    UP_PROB = 0.25
    LEFT_PROB = 0.25
    RIGHT_PROB = 0.25
    DOWN_PROB = 0.25

    def __init__(self):
        super(MyPrey, self).__init__(WildLifeAgentData.PREY)


    def response(self, perceptions):
        """
        :param perceptions: The perceptions of the agent at each step
        :return: The `Action' that your agent takes after perceiving the environment at each step
        """
        agent_pos = perceptions.agent_position
        probability_map = ProbabilityMap()
        probability_map.put(MyAction.NORTH,  MyPrey.UP_PROB)
        probability_map.put(MyAction.SOUTH, MyPrey.DOWN_PROB)
        probability_map.put(MyAction.WEST, MyPrey.LEFT_PROB)
        probability_map.put(MyAction.EAST, MyPrey.RIGHT_PROB)

        for obstacle_pos in perceptions.obstacles:
            if agent_pos.get_distance_to(obstacle_pos) > 1:
                continue

            relative_orientation = agent_pos.get_simple_relative_orientation(obstacle_pos)
            if relative_orientation == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.NORTH)

            elif relative_orientation == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.SOUTH)

            elif relative_orientation == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.EAST)

            elif relative_orientation == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.WEST)

        ## save available moves
        available_moves = ProbabilityMap(existing_map=probability_map)

        ## examine actions which are unavailable because of predators
        for (_, predator_pos) in perceptions.nearby_predators:
            relative_pos = agent_pos.get_simple_relative_orientation(predator_pos)

            if relative_pos == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.NORTH)

            elif relative_pos == GridRelativeOrientation.FRONT_LEFT:
                probability_map.remove(MyAction.NORTH)
                probability_map.remove(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.FRONT_RIGHT:
                probability_map.remove(MyAction.NORTH)
                probability_map.remove(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.SOUTH)

            elif relative_pos == GridRelativeOrientation.BACK_LEFT:
                probability_map.remove(MyAction.SOUTH)
                probability_map.remove(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.BACK_RIGHT:
                probability_map.remove(MyAction.SOUTH)
                probability_map.remove(MyAction.EAST)

        if not probability_map.empty():
            return probability_map.choice()
        else:
            return available_moves.choice()



class MyPredator(WildLifeAgent):

    def __init__(self, map_width=None, map_height=None):
        super(MyPredator, self).__init__(WildLifeAgentData.PREDATOR)
        self.map_width = map_width
        self.map_height = map_height

    def response(self, perceptions: MyAgentPerception) -> MyAction:
        """
        TODO your response function for the predator agent with NO communication
        :param perceptions:
        :return:
        """
        agent_pos = perceptions.agent_position
        probability_map = ProbabilityMap()
        probability_map.put(MyAction.NORTH,  MyPrey.UP_PROB)
        probability_map.put(MyAction.SOUTH, MyPrey.DOWN_PROB)
        probability_map.put(MyAction.WEST, MyPrey.LEFT_PROB)
        probability_map.put(MyAction.EAST, MyPrey.RIGHT_PROB)

        for obstacle_pos in perceptions.obstacles:
            if agent_pos.get_distance_to(obstacle_pos) > 1:
                continue

            relative_orientation = agent_pos.get_simple_relative_orientation(obstacle_pos)
            if relative_orientation == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.NORTH)

            elif relative_orientation == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.SOUTH)

            elif relative_orientation == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.EAST)

            elif relative_orientation == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.WEST)

        ## save available moves
        available_moves = ProbabilityMap(existing_map=probability_map)

        ## examine actions which should be followed to catch the prey
        ## this means that if the prey is in one of the 8 directions, the predator should remove the opposite direction
        for (_, prey_pos) in perceptions.nearby_prey:
            relative_pos = agent_pos.get_simple_relative_orientation(prey_pos)

            if relative_pos == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.SOUTH)

            elif relative_pos == GridRelativeOrientation.FRONT_LEFT:
                probability_map.remove(MyAction.SOUTH)
                probability_map.remove(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.FRONT_RIGHT:
                probability_map.remove(MyAction.SOUTH)
                probability_map.remove(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.NORTH)

            elif relative_pos == GridRelativeOrientation.BACK_LEFT:
                probability_map.remove(MyAction.NORTH)
                probability_map.remove(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.BACK_RIGHT:
                probability_map.remove(MyAction.NORTH)
                probability_map.remove(MyAction.WEST)

        if not probability_map.empty():
            return probability_map.choice()
        else:
            return available_moves.choice()
        

class MyPredatorWithCommunication(MyPredator):

    def __init__(self, map_width=None, map_height=None):
        super(MyPredatorWithCommunication, self).__init__(map_width, map_height)

    def response(self, perceptions: MyAgentPerception) -> SocialAction:
        """
        TODO your response function for the predator agent WITH communication
        :param perceptions:
        :return:
        """
        return SocialAction(random.choice([MyAction.NORTH, MyAction.SOUTH, MyAction.EAST, MyAction.WEST]))



class MyEnvironment(HuntingEnvironment):
    """
    Your implementation of the environment in which cleaner agents work.
    """
    PREY_RANGE = 2
    PREDATOR_RANGE = 3

    def __init__(self, predator_agent_type, w, h, num_predators, num_prey, rand_seed = 42, prey_kill_times = None):
        """
        Default constructor. This should call the initialize methods offered by the super class.
        """
        if not prey_kill_times:
            self.prey_kill_times = []
        else:
            self.prey_kill_times = prey_kill_times
        self.step_count = 0

        print("Seed = %i" % rand_seed)
        super(MyEnvironment, self).__init__()

        predators = []
        prey = []

        for i in range(num_predators):
            predators.append(predator_agent_type(map_width=w, map_height=h))

        for i in range(num_prey):
            prey.append(MyPrey())

        """ Message box for messages that need to be delivered by the environment to their respective recepients, on
        the next turn """
        self.message_box = []

        ## initialize the huniting environment
        self.initialize(w=w, h=h, predator_agents=predators, prey_agents=prey, rand_seed = rand_seed)


    def step(self):
        """
        This method should iterate through all agents, provide them provide them with perceptions, and apply the
        action they return.
        """
        """
        STAGE 1: generate perceptions for all agents, based on the state of the environment at the beginning of this
        turn
        """
        agent_perceptions = {}

        ## get perceptions for prey agents
        for prey_data in self._prey_agents:
            nearby_obstacles = self.get_nearby_obstacles(prey_data.grid_position, MyEnvironment.PREY_RANGE)
            nearby_predators = self.get_nearby_predators(prey_data.grid_position, MyEnvironment.PREY_RANGE)
            nearby_prey = self.get_nearby_prey(prey_data.grid_position, MyEnvironment.PREY_RANGE)

            predators = [(ag_data.linked_agent.id, ag_data.grid_position) for ag_data in nearby_predators]
            prey = [(ag_data.linked_agent.id, ag_data.grid_position) for ag_data in nearby_prey]

            agent_perceptions[prey_data] = MyAgentPerception(agent_position=prey_data.grid_position,
                                                             obstacles=nearby_obstacles,
                                                             nearby_predators=predators,
                                                             nearby_prey=prey)

        ## TODO: create perceptions for predator agents, including messages in the `message_box`
        for predator_data in self._predator_agents:
            nearby_obstacles = self.get_nearby_obstacles(predator_data.grid_position, MyEnvironment.PREDATOR_RANGE)
            nearby_predators = self.get_nearby_predators(predator_data.grid_position, MyEnvironment.PREDATOR_RANGE)
            nearby_prey = self.get_nearby_prey(predator_data.grid_position, MyEnvironment.PREDATOR_RANGE)

            predators = [(ag_data.linked_agent.id, ag_data.grid_position) for ag_data in nearby_predators]
            prey = [(ag_data.linked_agent.id, ag_data.grid_position) for ag_data in nearby_prey]

            agent_perceptions[predator_data] = MyAgentPerception(agent_position=predator_data.grid_position,
                                                                 obstacles=nearby_obstacles,
                                                                 nearby_predators=predators,
                                                                 nearby_prey=prey,
                                                                 messages=AgentMessage.filter_messages_for(self.message_box, predator_data.linked_agent))
        
        """
        STAGE 2: call response for each agent to obtain desired actions
        """
        agent_actions = {}
        ## TODO: get actions for all agents
        for prey_data in self._prey_agents:
            agent_actions[prey_data] = prey_data.linked_agent.response(agent_perceptions[prey_data])

        for predator_data in self._predator_agents:
            agent_actions[predator_data] = predator_data.linked_agent.response(agent_perceptions[predator_data])

        """
        STAGE 3: apply the agents' actions in the environment
        """
        for prey_data in self._prey_agents:
            if not prey_data in agent_actions:
                print("Agent %s did not opt for any action!" % str(prey_data))

            else:
                prey_action = agent_actions[prey_data]
                new_position = None

                if prey_action == MyAction.NORTH:
                    new_position = prey_data.grid_position.get_neighbour_position(GridOrientation.NORTH)
                elif prey_action == MyAction.SOUTH:
                    new_position = prey_data.grid_position.get_neighbour_position(GridOrientation.SOUTH)
                elif prey_action == MyAction.EAST:
                    new_position = prey_data.grid_position.get_neighbour_position(GridOrientation.EAST)
                elif prey_action == MyAction.WEST:
                    new_position = prey_data.grid_position.get_neighbour_position(GridOrientation.WEST)

                if not new_position in self._xtiles:
                    prey_data.grid_position = new_position
                else:
                    print("Agent %s tried to go through a wall!" % str(prey_data))

        for predator_data in self._predator_agents:
            if not predator_data in agent_actions:
                print("Agent %s did not opt for any action!" % str(predator_data))

            else:
                predator_action = agent_actions[predator_data]
                new_position = None
                ## TODO: handle case for a SocialAction instance

                if predator_action == MyAction.NORTH:
                    new_position = predator_data.grid_position.get_neighbour_position(GridOrientation.NORTH)
                elif predator_action == MyAction.SOUTH:
                    new_position = predator_data.grid_position.get_neighbour_position(GridOrientation.SOUTH)
                elif predator_action == MyAction.EAST:
                    new_position = predator_data.grid_position.get_neighbour_position(GridOrientation.EAST)
                elif predator_action == MyAction.WEST:
                    new_position = predator_data.grid_position.get_neighbour_position(GridOrientation.WEST)

                if not new_position in self._xtiles:
                    predator_data.grid_position = new_position
                else:
                    print("Agent %s tried to go through a wall!" % str(predator_data))


        # increment the step count
        self.step_count += 1

        """
        At the end of the turn remove the dead prey. If any prey was killed, add a tuple containing the 
        current step count and the number of prey killed at this step to the list of prey kill times.
        """
        num_prey_killed = self.remove_dead_prey()
        if num_prey_killed > 0:
            self.prey_kill_times.append((self.step_count, num_prey_killed))


    def get_step_count(self):
        """
        :return: the number of steps that have been executed in the environment
        """
        return self.step_count
    
    def get_prey_kill_times(self):
        """
        :return: a list of tuples containing the step count and the number of prey killed at that step
        """
        return self.prey_kill_times


class Tester(object):

    def __init__(self, predator_agent_type = MyPredator, num_predators=4, num_prey=10, width=15, height=10, rand_seed = 42, delay=0.1):
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.width = width
        self.height = height
        self.delay = delay

        # reset the agent counter for generating unique agent ids
        WildLifeAgent.agent_counter = 0

        self.env = MyEnvironment(predator_agent_type, self.width, self.height, self.num_predators, self.num_prey, rand_seed=rand_seed)
        self.make_steps()

    def make_steps(self):
        while not self.env.goals_completed():
            self.env.step()

            print(self.env)

            time.sleep(self.delay)
        
        # return the number of steps and the prey kill times
        return self.env.get_step_count(), self.env.get_prey_kill_times()


if __name__ == "__main__":
    tester = Tester(predator_agent_type=MyPredator, rand_seed=42, delay=0.1)
    step_count, prey_kill_times = tester.make_steps()
    print("Step count: ", step_count)
    print("Prey kill times: ", prey_kill_times)

    # NUM_TESTS = 20
    
    # step_count_list = []
    # prey_kill_times_list = []

    # for i in range(NUM_TESTS):
    #     tester = Tester(predator_agent_type=MyPredator, rand_seed=42+i, delay=0.1)
    #     step_count, prey_kill_times = tester.make_steps()

    #     step_count_list.append(step_count)
    #     prey_kill_times_list.append(prey_kill_times)

    # # Make an analysis of the min, max, median step counts and standard deviation as a describe call
    # print("Step count analysis")
    # print(pd.Series(step_count_list).describe())

    # # Make an analysis of the most common kill times as a scatter plot
    # print("Prey kill times analysis")
    # prey_kill_times = [item for sublist in prey_kill_times_list for item in sublist]
    # df = pd.DataFrame(prey_kill_times, columns=["Step", "Prey killed"])
    # df.plot(kind="scatter", x="Step", y="Prey killed", xlabel="Step", ylabel="Prey killed", yticks=range(0, 11, 1))
    # plt.show()

