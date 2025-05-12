from typing import Callable, List
from commons import CommonsAgent, CommonsPerception
from communication import AgentAction
import numpy as np

class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.last_utilities = []
        self.round_shares = []
        self.resource_history = []
        self.cycle_offset = agent_id % 3

    def specify_share(self, perception: CommonsPerception) -> float:
        num_agents = perception.num_agents
        fair_share = 1.0 / num_agents
        base_share = 0.9 * fair_share

        self.resource_history.append(perception.resource_remaining)
        recent_resource_trend = 0
        if len(self.resource_history) >= 3:
            recent_resource_trend = self.resource_history[-1] - self.resource_history[-3]

        round_idx = len(self.round_shares)
        if round_idx % 3 == self.cycle_offset:
            base_share *= 1.2
        else:
            base_share *= 0.9

        if perception.resource_remaining < 0.2 * num_agents or recent_resource_trend < -10:
            base_share *= 0.5
        elif perception.resource_remaining < 0.5 * num_agents:
            base_share *= 0.7

        if self.last_utilities:
            avg_util = sum(self.last_utilities[-3:]) / min(3, len(self.last_utilities))
            if avg_util < 2.0:
                base_share *= 1.1
            elif avg_util > 5.0:
                base_share *= 0.9

        share = min(base_share, fair_share)
        self.round_shares.append(share)
        return share

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        my_share = perception.resource_shares[self.id]
        all_shares = list(perception.resource_shares.values())
        avg_share = np.mean(all_shares)
        total_share = sum(all_shares)
        adjustments = {}

        if perception.resource_remaining < 0.2 * perception.num_agents:
            for agent_id in perception.resource_shares:
                adjustments[agent_id] = -0.03
            new_share = max(0.0, my_share - 0.03)
            return AgentAction(self.id, resource_share=new_share, consumption_adjustment=adjustments)

        if total_share >= 0.85:
            for agent_id, share in perception.resource_shares.items():
                if share > avg_share * 1.1:
                    adjustments[agent_id] = -0.02
            new_share = max(0.0, my_share - 0.02) if my_share > avg_share else my_share
            return AgentAction(self.id, resource_share=new_share, consumption_adjustment=adjustments)

        if total_share < 0.8 and my_share < avg_share * 0.8:
            new_share = min(my_share + 0.01, 0.05)
            return AgentAction(self.id, resource_share=new_share)

        return AgentAction(self.id, resource_share=my_share, no_action=True)

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        if perception.agent_utilities and self.id in perception.agent_utilities:
            self.last_utilities.append(perception.agent_utilities[self.id])
