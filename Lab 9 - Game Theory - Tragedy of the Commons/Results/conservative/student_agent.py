from typing import Callable, List
from commons import CommonsAgent, CommonsPerception
from communication import AgentAction
import numpy as np

class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.last_utilities = []         # Stores the agent's recent utilities
        self.round_shares = []           # Stores history of shares requested each round
        self.resource_history = []       # Stores remaining resources per round

    def specify_share(self, perception: CommonsPerception) -> float:
        num_agents = perception.num_agents
        base_share = 0.9 / num_agents    # Conservative default share slightly below fair share

        self.resource_history.append(perception.resource_remaining)

        # Calculate the trend in available resources over the last 3 rounds
        recent_resource_trend = 0
        if len(self.resource_history) >= 3:
            recent_resource_trend = self.resource_history[-1] - self.resource_history[-3]

        # If resources are critically low or rapidly depleting, reduce share significantly
        if perception.resource_remaining < 0.2 * num_agents or recent_resource_trend < -10:
            base_share *= 0.5
        # If moderately low, apply a moderate reduction
        elif perception.resource_remaining < 0.5 * num_agents:
            base_share *= 0.7

        # If the agent had poor recent utility, it becomes more conservative
        if self.last_utilities:
            avg_util = sum(self.last_utilities[-3:]) / min(3, len(self.last_utilities))
            if avg_util < 1.0:
                base_share *= 0.85

        # If resources are abundant and increasing, be slightly more aggressive
        if recent_resource_trend > 10 and perception.resource_remaining > 5 * num_agents:
            base_share *= 1.1

        # Final limit: don't exceed the fair share
        share = min(base_share, 1.0 / num_agents)
        self.round_shares.append(share)
        return share

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        my_share = perception.resource_shares[self.id]
        all_shares = list(perception.resource_shares.values())
        avg_share = np.mean(all_shares)
        total_share = sum(all_shares)
        adjustments = {}

        # If resources are critically low, all agents are penalized equally
        if perception.resource_remaining < 0.2 * perception.num_agents:
            for agent_id in perception.resource_shares:
                adjustments[agent_id] = -0.03
            new_share = max(0.0, my_share - 0.03)
            return AgentAction(self.id, resource_share=new_share, consumption_adjustment=adjustments)

        # If total requested resources is too high, penalize agents who overconsume
        if total_share >= 0.85:
            for agent_id, share in perception.resource_shares.items():
                if share > avg_share * 1.1:
                    adjustments[agent_id] = -0.02
            new_share = max(0.0, my_share - 0.02) if my_share > avg_share else my_share
            return AgentAction(self.id, resource_share=new_share, consumption_adjustment=adjustments)

        # If total requested resources is too low and the agent is under-consuming, increase share slightly
        if total_share < 0.8 and my_share < avg_share * 0.8:
            new_share = min(my_share + 0.01, 0.05)
            return AgentAction(self.id, resource_share=new_share)

        # Otherwise, maintain the current share
        return AgentAction(self.id, resource_share=my_share, no_action=True)

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        # Track utility after each round to adjust behavior in the future
        if perception.agent_utilities and self.id in perception.agent_utilities:
            self.last_utilities.append(perception.agent_utilities[self.id])
