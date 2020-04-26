# Robo-Daycare

Humans are capable of learning new tasks very quickly by leveraging their experience. Our innate intelligence allows us to recognize objects from very few examples and learnto complete a new foreign task with very little experience.
Our initial goal is to teach a robot how to walk using the Proximal Policy Optimization algorithm and then use it as a baseline for various other transfer learning tasks like playing soccer.
During the course of this, we found that transfer learning is a reliable method to layer behaviors on top of each other in a way that would be necessary for complex tasks like playing soccer.

## Setting Up

First, you can perform a minimal installation of OpenAI Gym with
```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

Then, Pybullet-Gym is to be installed. Clone the repository and install locally
```bash
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

Important Note: *Do not* use `python setup.py install` as this will not copy the assets (you might get missing SDF file errors).

To test installation, open python and run
```python
import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym

env = gym.make('HumanoidPyBulletEnv-v0')
# env.render() # call this before env.reset, if you want a window showing the environment
env.reset()  # should return a state vector if everything worked
```

## State of implementations

Environment Name |
---------|
| **New Environments** |
Walker2DWindPyBulletEnv-v0		|
HopperWindPyBulletEnv-v0		|
HumanoidWindPyBulletEnv-v0		|
Walker2DDribblePyBulletEnv-v0	|
Walker2DKickPyBulletEnv-v0		|
Walker2DSoccerPyBulletEnv-v0	|