# Reinforcement Learning Showcase
## Abstract
This purpose of this project is to **showcase different reinforcement 
learning algorithms**.
For this project the real-time strategy game *Starcraft* was chosen as a 
simulation environment. Its game functionalities are exposed through an API, which is used by the Python package *pysc2* developed by *DeepMind*.

Because the complete Starcraft game is complex, the Starcraft API 
divides the game into multiple minigames, which allows to tackle individual 
game elements separately. Minigames are very focused scenarios on small maps 
to learn/test actions and game mechanics.

The available minigames in Starcraft are:
* **MoveToBeacon**
* CollectMineralShards
* FindAndDefeatZerglings
* DefeatRoaches
* DefeatZerglingsAndBenelings
* CollectMineralsAndGas
* BuildMarines

In this project, the different reinforcement learning algorithms are tested 
only on the minigame **MoveToBeacon**. Within this game the task of the agent ("marine") is to navigate to a target ("beacon") in the most efficient way
possible.

## Installation
The installation process is documented [here](docs/Installation.md).

## Environment
All information related to the environment is documented 
[here](docs/Environment.md).

## Architecture
General design decisions are documented [here](docs/Architecture.md)

## Agents
The behaviour of the marine is encapsulated in the *Agent* class. So 
additional to a *RandomAgent* (with random behaviour) and a *BasicAgent* 
(with optimal behaviour) all learning algorithms have a separate *Agent* class.

The different agents are documented [here](docs/Agents.md).

* [Learning algorithms](docs/Agents.md)
* [Exploration Strategies](docs/ExplorationStrategies.md)