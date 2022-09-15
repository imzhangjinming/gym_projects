import argparse
import sys, os
import time
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import datetime
from env import NormalizedActions, OUNoise
from ddpg import DDPG
from common.utils import make_dir, save_results
import gym

def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
    parser = argparse.ArgumentParser(description="hyperparamters")
    parser.add_argument('--algo-name', default="DDPG", type=str, help="name of algorithm")
    parser.add_argument('--env_name',default='Pendulum-v1',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=300,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--critic_lr',default=1e-3,type=float,help="learning rate of critic")
    parser.add_argument('--actor_lr',default=1e-4,type=float,help="learning rate of actor")
    parser.add_argument('--buffer_capacity',default=8000,type=int,help="replay buffer capacity")
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--soft_tau',default=5e-2,type=float)
    parser.add_argument('--hidden_size',default=512,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) # path to save models
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")   
    args = parser.parse_args()                           
    return args 

def env_agent_config(cfg,seed=1):
    env = NormalizedActions(gym.make(cfg.env_name)) # 装饰action噪声
    env.seed(seed) # 随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states,n_actions,cfg)
    return env, agent

def train(cfg, env, agent):
    print("START TRAINING")
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    ounoise = OUNoise(env.action_space)
    rewards = []
    ma_rewards = [] # move average rewards
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ounoise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = ounoise.get_action(action, i_step)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        if (i_ep+1)%10==0:
            print(f'Env:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print("FINISH TRAINING")
    return {'rewards': rewards, 'ma_rewards': ma_rewards}

def test(cfg, env, agent):
    print("START TESTING")
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')  
    rewards = []
    ma_rewards = [] # move average rewards 
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            time.sleep(0.1)
            ep_reward += reward
            # agent.buffer.push(state, action, reward, next_state, done)
            # agent.update()
            state = next_state
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"Epside:{i_ep+1}/{cfg.test_eps}, Reward:{ep_reward:.1f}")
    print("FINISH TESTING")
    return {'rewards': rewards, 'ma_rewards': ma_rewards}

if __name__ == "__main__":
    cfg = get_args()
    # training
    env,agent = env_agent_config(cfg,seed=1)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    # save_args(cfg, cfg.result_path)
    agent.save(path=cfg.model_path)
    save_results(res_dic, tag='train',
                 path=cfg.result_path)  
    # plot_rewards(res_dic['rewards'], cfg, tag="train") # , res_dic['ma_rewards']
    # testing
    env,agent = env_agent_config(cfg,seed=0)
    agent.load(path=cfg.model_path)
    res_dic = test(cfg,env,agent)
    save_results(res_dic, tag='test',
                 path=cfg.result_path)  
    # plot_rewards(res_dic['rewards'], cfg, tag="test")  # , res_dic['ma_rewards']