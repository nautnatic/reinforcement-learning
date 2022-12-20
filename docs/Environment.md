# Environment
The agent operates in a 2D rectangle environment. The environment doesn't 
contain obstacles or agents other than the marine and the beacon. Each time 
the marine reaches the beacon, the beacon gets initialized in a new position.

The *reward* equals to the number of times that the agent reached the beacon within a episode. The end of an episode is reached, when a specified number of simulation steps is reached.

The *observation space* is limited to what a human player knows as well. It 
consists of:
...

The interface to the environment consists of observation space and action 
space. 
The observation space consists of all information available about the 
environment. The action space consists of all actions available to influence 
the environment.
