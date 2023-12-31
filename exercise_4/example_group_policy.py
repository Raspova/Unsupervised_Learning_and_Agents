
from agent import Agent
import random

def default_policy(agent: Agent):
    print(agent.known_rewards, agent.position, "\n")
    if agent.known_rewards[agent.position] > 0:
        return "none"
    if agent.position <= 0:
        return "right"
    if agent.position >= 8:
        return "left"
    if random.random() < 0.5:
        return "right"
    else:
        return "left"