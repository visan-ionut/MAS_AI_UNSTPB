from environment import *
from typing import List, Tuple
import time

class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, target_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)

        self.target_state = target_state

        """
        The agent's belief about the world state. Initially, the agent has no belief about the world state.
        """
        self.belief: BlocksWorld = None

        """
        The agent's current desire. It is expressed as a list of blocks for which the agent wants to make a plan to bring to their corresponding
        configuration in the target state.
        The list can contain a single block or a sequence of blocks that represent: (i) a stack of blocks, (ii) a row of blocks (e.g. going level by level).
        """
        self.current_desire : List[Block] = None

        """
        The current intention is the agent plan (sequence of actions) that the agent is executing to achieve the current desire.
        """
        self.current_intention: List[BlocksWorldAction] = []


    def response(self, perception: BlocksWorldPerception) -> BlocksWorldAction:
        ## if the perceived state contains the target state, the agent has achieved its goal
        if perception.current_world.contains_world(self.target_state):
            return AgentCompleted()

        ## revise the agents beliefs based on the perceived state
        self.revise_beliefs(perception.current_world, perception.previous_action_succeeded, perception)

        ## Single minded agent intention execution: if the agent still has actions left in the current intention, and the intention
        ## is still applicable to the perceived state, the agent continues executing the intention
        if len(self.current_intention) > 0 and self._can_apply_action(self.current_intention[0], perception.current_world, perception.holding_block):
            return self.current_intention.pop(0)
        else:
            ## the agent has to set a new current desire and plan to achieve it
            self.current_desire, self.current_intention = self.plan()

        ## If there is an action in the current intention, pop it and return it
        if len(self.current_intention) > 0:
            return self.current_intention.pop(0)
        else:
            ## If there is no action in the current intention, return a NoAction
            return NoAction()



    def _can_apply_action(self, act: BlocksWorldAction, world: BlocksWorld, holding_block: str) -> bool:
        """
        Check if the action can be applied to the current world state.
        """
        ## create a clone of the world
        sim_world = world.clone()

        ## apply the action to the clone, surrpressing any exceptions
        try:
            ## locking can be performed at any time, so check if the action is a lock actio
            if act.get_type() == "lock":
                ## try to lock the block
                sim_world.lock(act.get_argument())
            else:
                if holding_block is None:
                    if act.get_type() == "putdown" or act.get_type() == "stack":
                        ## If we are not holding anything, we cannot putdown or stack a block
                        return False

                    if act.get_type() == "pickup":
                        ## try to pickup the block
                        sim_world.pickup(act.get_argument())
                    elif act.get_type() == "unstack":
                        ## try to unstack the block
                        sim_world.unstack(act.get_first_arg(), act.get_second_arg())
                else:
                    ## we are holding a block, so we can only putdown or stack
                    if act.get_type() == "pickup" or act.get_type() == "unstack":
                        ## If we are holding a block, we cannot pickup or unstack
                        return False

                    if act.get_type() == "putdown":
                        ## If we want to putdown the block we have to check if it's the same block we are holding
                        if act.get_argument() != holding_block:
                            return False

                    if act.get_type() == "stack":
                        ## If we want to stack the block we have to check if it's the same block we are holding
                        if act.get_first_arg() != holding_block:
                            return False
                        ## try to stack the block
                        sim_world.stack(act.get_first_arg(), act.get_second_arg())
        except Exception as e:
            return False

        return True


    def revise_beliefs(self, perceived_world_state: BlocksWorld, previous_action_succeeded: bool, perception: BlocksWorldPerception):
        print("\nUpdating beliefs...")

        # Update internal belief and held block from perception
        self.belief = perceived_world_state.clone()
        self.held_block = perception.holding_block
        print(f"Block held by agent: {self.held_block}" if self.held_block else "Agent is not holding any block.")

        # Get all blocks currently present in the perceived world
        self.active_blocks = set()
        for stack in self.belief.get_stacks():
            self.active_blocks.update(stack.get_blocks())

        print(f"Active blocks in the world: {self.active_blocks}")

        # Identify blocks needed for the goal but not currently present
        required_blocks = set(self.target_state.get_all_blocks())
        self.stash_blocks = required_blocks - self.active_blocks

        # Remove held block from stash if already in hand
        if self.held_block and self.held_block in self.stash_blocks:
            self.stash_blocks.remove(self.held_block)

        print(f"Blocks in stash: {self.stash_blocks}" if self.stash_blocks else "Stash is empty.")

        # Identify missing blocks (still needed)
        self.missing_blocks = required_blocks - self.active_blocks
        if self.held_block and self.held_block in self.stash_blocks:
            self.stash_blocks.remove(self.held_block)

        print(f"Missing blocks: {self.missing_blocks}")

        # Display current perceived stack structure
        print("Current stack structure in agent's perception:")
        for stack in self.belief.get_stacks():
            print(f"Stack: {stack.get_blocks()}")

        # Extract desired stack configuration and base blocks
        self.desired_stacks = [stack.get_blocks() for stack in self.target_state.get_stacks()]
        self.base_blocks = [stack[0] for stack in self.desired_stacks if stack]

        print(f"Target stacks: {self.desired_stacks}")
        print(f"Base blocks of stacks: {self.base_blocks}")

        print("\nComparing current world state with desired state...")

        current_positions = {}
        target_positions = {}

        # Helper: remove duplicate blocks in a stack (if any)
        def remove_locked_duplicates(stack):
            cleaned_stack = []
            seen_blocks = set()
            for block in stack:
                if block not in seen_blocks:
                    cleaned_stack.append(block)
                    seen_blocks.add(block)
            return cleaned_stack

        # Build current block positions
        for stack in self.belief.get_stacks():
            stack_blocks = remove_locked_duplicates(stack.get_blocks())
            for i, block in enumerate(stack_blocks):
                current_positions[block] = (i, stack_blocks)

        # Build target block positions
        for stack in self.target_state.get_stacks():
            stack_blocks = stack.get_blocks()
            for i, block in enumerate(stack_blocks):
                target_positions[block] = (i, stack_blocks)

        misplaced_blocks = []

        # Compare block positions and find misplaced ones
        for block in self.active_blocks:
            current_info = current_positions.get(block, None)
            target_info = target_positions.get(block, None)

            if not current_info:
                print(f"Block {block} is missing from the current environment!")
                continue

            if not target_info:
                print(f"Block {block} is not part of the target state!")
                continue

            current_index, current_stack = current_info
            target_index, target_stack = target_info

            if current_stack == target_stack and current_index == target_index:
                print(f"✅ Block {block} is correctly placed.")
            else:
                print(f"❌ Block {block} is misplaced! "
                      f"Current: Stack {current_stack}, Position {current_index} | "
                      f"Target: Stack {target_stack}, Position {target_index}")
                misplaced_blocks.append(block)

        print(f"\nTotal misplaced blocks: {len(misplaced_blocks)} - {misplaced_blocks}")

        self.misplaced_blocks = misplaced_blocks

        # If last action failed unexpectedly, reset plan
        if not previous_action_succeeded:
            print("! Unexpected failure detected! Resetting intentions.")
            self.current_intention = []
        else:
            print("Last action was successful.")


    def plan(self) -> Tuple[List[Block], List[BlocksWorldAction]]:
        print("Agent is making a plan...")

        actions = []
        misplaced_blocks = self.misplaced_blocks
        correct_blocks = self.active_blocks - set(misplaced_blocks)

        print(f"Blocks correctly placed: {correct_blocks}")
        print(f"Blocks misplaced: {misplaced_blocks}")

        if not misplaced_blocks:
            print("All blocks are correctly placed! Agent completes the task.")
            return [], [AgentCompleted()]

        # STEP 1: If the agent is holding a block, try placing it
        if self.held_block:
            print(f"Agent is holding {self.held_block}, deciding where to place it...")

            for stack in self.desired_stacks:
                for i in range(len(stack) - 1):
                    block = stack[i + 1]
                    below_block = stack[i]

                    if block == self.held_block:
                        if not self.belief.exists(below_block) or below_block in self.stash_blocks:
                            print(f"! Cannot place {block} on {below_block}, base missing. Checking other moves...")
                            continue

                        try:
                            stack_data = self.belief.get_stack(below_block)
                            if stack_data.get_top_block() != below_block:
                                print(f"! Cannot place {block} on {below_block}, not topmost. Checking other moves...")
                                continue
                        except ValueError:
                            print(f"! Cannot place {block} on {below_block}, stack error. Checking other moves...")
                            continue

                        print(f"Placing {block} on {below_block}.")
                        return [], [Stack(block, below_block)]

            print(f"Placing {self.held_block} on the table.")
            return [], [PutDown(self.held_block)]

        # STEP 2: Make sure base blocks are in position and locked
        for base_block in self.base_blocks:
            if not self.belief.exists(base_block) or base_block in self.stash_blocks:
                print(f"Base block {base_block} is missing! Checking other possible moves...")
                continue

            if self.belief.is_on_table(base_block):
                try:
                    base_stack = self.belief.get_stack(base_block)
                    if not base_stack.is_locked(base_block):
                        print(f"Locking base block: {base_block}")
                        actions.append(Lock(base_block))
                except ValueError:
                    print(f"Error accessing {base_block}. Skipping lock.")
                    continue

        # STEP 3: Move misplaced blocks to the table
        moved_blocks = False
        for block in self.active_blocks:
            if not self.belief.exists(block):
                continue

            try:
                stack = self.belief.get_stack(block)
            except ValueError:
                continue

            if stack.is_locked(block):
                continue

            if not self.belief.is_on_table(block):
                if stack.get_top_block() == block:
                    below_block = stack.get_below(block)
                    print(f"Moving block {block} to the table.")
                    actions.append(Unstack(block, below_block))
                    actions.append(PutDown(block))
                    moved_blocks = True

        if actions:
            print("Executing useful moves before waiting.")
            return [], actions

        # STEP 4: Build the desired stacks in correct order
        for stack in self.desired_stacks:
            for i in range(len(stack) - 1):
                block = stack[i + 1]
                below_block = stack[i]

                if not self.belief.exists(block) or block in self.stash_blocks:
                    print(f"! Block {block} missing. Checking alternatives before waiting...")
                    continue

                if not self.belief.exists(below_block) or below_block in self.stash_blocks:
                    print(f"! Block {below_block} missing. Checking alternatives before waiting...")
                    continue

                try:
                    stack_data = self.belief.get_stack(block)
                except ValueError:
                    print(f"! Block {block} is no longer in a stack. Checking alternatives...")
                    continue

                if stack_data.get_below(block) == below_block:
                    if not stack_data.is_locked(block):
                        print(f"Block {block} is already in position. Locking it.")
                        actions.append(Lock(block))
                    continue

                # Check if we need to unstack before picking up
                if not self.belief.is_on_table(block):
                    top_block = stack_data.get_top_block()
                    if top_block != block:
                        print(f"Block {block} is not at the top, unstacking {top_block}.")
                        actions.append(Unstack(top_block, stack_data.get_below(top_block)))
                        actions.append(PutDown(top_block))
                        continue

                print(f"Moving block {block} onto {below_block}.")
                actions.append(PickUp(block))
                actions.append(Stack(block, below_block))
                actions.append(Lock(block))

        if actions:
            return [], actions

        print("No more useful moves. Waiting...")
        return [], [NoAction()]


    def status_string(self):
        # TODO: return information about the agent's current state and current plan.
        return str(self) + ": Hai agentule, pune si tu blocurile alea."



class Tester(object):
    STEP_DELAY = 0.5
    TEST_SUITE = "tests/0e-large/"

    EXT = ".txt"
    SI  = "si"
    SF  = "sf"

    DYNAMICS_PROB = .5

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)



    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(BlocksWorld(input_stream=input_stream))


    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            print("Agent %s desires:" % str(agent))
            print(str(desires))


    def make_steps(self):
        print("\n\n================================================= INITIAL STATE:")
        print(str(self._environment))
        print("\n\n=================================================")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            print(str(self._environment))

            for ag in self._agents:
                print(ag.status_string())

            nr_steps += 1

            print("\n\n================================================= STEP %i completed." % nr_steps)

        print("\n\n================================================= ALL STEPS COMPLETED")



if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()
