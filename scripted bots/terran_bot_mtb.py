#terran bot

from pysc2.agents import scripted_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app

class TerranMTB_Agent(scripted_agent.MoveToBeacon):
  def step(self, obs):
    super(TerranMTB_Agent, self).step(obs)
    
    return actions.FUNCTIONS.no_op()

def main(unused_argv):
  agent = scripted_agent.MoveToBeacon()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="MoveToBeacon",
          players=[sc2_env.Agent(sc2_env.Race.terran)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64)),
          step_mul=16,
          game_steps_per_episode=0,
          visualize=True,
          realtime=True) as env:
          
        agent.setup(env.observation_spec(), env.action_spec())
        
        timesteps = env.reset()
        agent.reset()
        
        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
      
  except KeyboardInterrupt:
    pass
  
if __name__ == "__main__":
  app.run(main)