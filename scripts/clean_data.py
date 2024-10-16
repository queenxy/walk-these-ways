import numpy as np

data = np.load("trotting.npz")
print(data["states"].shape)
print(data["actions"].shape)
print(data["rews"].shape)
print(data["dones"].shape)
o = data["states"]
a = data["actions"]
r = data["rews"]
d = data["dones"]

def get_obs(obs):
    o = np.zeros(1170)
    for i in range(30):
        o[i*39:(i+1)*39] = obs[i*70:i*70+39]
    return o

count = 0
fail_count = 0
obs_buf = []
act_buf = []
traj_lengths = []
for j in range(data["states"].shape[1]):
    tra_rew = 0
    tra_step = 0
    tra_obs = []
    tra_act = []
    for i in range(data["states"].shape[0]):
        obs = get_obs(o[i,j,:])
        action = a[i,j,:]
        rew = r[i,j,0]
        done = d[i,j,0]
        if done == 0:
            tra_rew += rew
            tra_step += 1
            tra_obs.append(obs)
            tra_act.append(action)
        elif done == 1:
            if tra_step > 500 and tra_rew > 3:
                obs_buf += tra_obs
                act_buf += tra_act
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

        else:
            print("Data Error")
            break

obs_buf = np.array(obs_buf)
act_buf = np.array(act_buf)
traj_lengths = np.array(traj_lengths)
print(count)
print(fail_count)
print(obs_buf.shape)
print(act_buf.shape)
print(np.sum(traj_lengths))

np.savez("trotting_clean.npz",states=obs_buf,actions=act_buf,images=None,traj_lengths=traj_lengths)