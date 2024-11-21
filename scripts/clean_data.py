import numpy as np

delay_steps = 0
history_steps = 30
obs_length = 42

data = np.load("bounding.npz")
print(data["states"].shape)
print(data["actions"].shape)
print(data["rews"].shape)
print(data["dones"].shape)
o = data["states"]
a = data["actions"]
r = data["rews"]
d = data["dones"]
# print(o[0,0,-70:])

def get_obs(obs):
    o = np.zeros(42)
    o[0:3] = obs[0:3]
    o[3:39] = obs[18:54]
    o[39:42] = [0., 0.5, 0.]
    return o

count = 0
fail_count = 0
obs_buf = []
act_buf = []
obs_history_buf = []
traj_lengths = []
for j in range(data["states"].shape[1]):
    tra_rew = 0
    tra_step = 0
    tra_obs = []
    tra_act = []
    obs_history = np.zeros((obs_length*history_steps))
    tra_obs_history = []
    for i in range(data["states"].shape[0]):
        if tra_step == 0:
            obs = get_obs(o[i,j,-70:])
            tra_obs += [obs] * delay_steps
            for k in range(delay_steps):
                obs_history = np.concatenate((obs_history[obs_length:], obs))
                tra_obs_history.append(obs_history) 
        obs = get_obs(o[i,j,-70:])
        obs_history = np.concatenate((obs_history[obs_length:], obs))
        action = a[i,j,:]
        action = np.clip(action, -10, 10)
        action *= 0.1
        rew = r[i,j,0]
        done = d[i,j,0]
        if done == 0:
            tra_rew += rew
            tra_step += 1
            # if count > 50 and count < 65:
            tra_obs.append(obs)
            tra_act.append(action)
            tra_obs_history.append(obs_history)
        elif done == 1:
            if tra_step > 500 and tra_rew > 3:
                # print(tra_obs[0][-39:])
                obs_buf += tra_obs #[:-delay_steps]
                act_buf += tra_act
                obs_history_buf += tra_obs_history #[:-delay_steps]
                # if count > 50 and count < 65:
                traj_lengths.append(tra_step)
                # print(tra_step)
                # print(tra_rew)
                # print(len(tra_obs))
                # print(len(tra_act))
                count += 1
            else:
                fail_count += 1
            tra_rew = 0
            tra_step = 0
            tra_obs = []
            tra_act = []
            obs_history = np.zeros((obs_length*history_steps))
            tra_obs_history = []

            if count >= 50:
                break

        else:
            print("Data Error")
            break

    if count >= 50:
                break

obs_buf = np.array(obs_buf)
act_buf = np.array(act_buf)
traj_lengths = np.array(traj_lengths)
obs_history_buf = np.array(obs_history_buf)
print(count)
print(fail_count)
print(obs_buf.shape)
print(act_buf.shape)
print(obs_history_buf.shape)
print(traj_lengths)
print(np.sum(traj_lengths))

np.savez("bounding_onehot.npz",states=obs_history_buf,actions=act_buf,images=None,traj_lengths=traj_lengths)