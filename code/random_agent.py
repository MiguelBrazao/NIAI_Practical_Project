
import marioai
import agents
import tasks.move_forward

def main():
    agent = agents.RandomAgent()
    task = tasks.move_forward.MoveForwardTask(visualization=True)
    exp = marioai.Experiment(task, agent)
    
    exp.max_fps = -1
    task.env.level_type = 0
    rewards = exp.doEpisodes()
    print(sum(rewards[0]))


if __name__ == '__main__':
    main()