import numpy as np

data1 = np.load("trotting_onehot.npz")
o1 = data1["states"]
a1 = data1["actions"]
t1 = data1["traj_lengths"]

data2 = np.load("pronking_onehot.npz")
o2 = data2["states"]
a2 = data2["actions"]
t2 = data2["traj_lengths"]

data3 = np.load("pacing_onehot.npz")
o3 = data3["states"]
a3 = data3["actions"]
t3 = data3["traj_lengths"]

data4 = np.load("bounding_onehot.npz")
o4 = data4["states"]
a4 = data4["actions"]
t4 = data4["traj_lengths"]

obs_buf = np.concatenate((o1,o2,o3,o4),axis=0)
act_buf = np.concatenate((a1,a2,a3,a4),axis=0)
traj_lengths = np.concatenate((t1,t2,t3,t4),axis=0)

print(obs_buf.shape)
print(act_buf.shape)
print(traj_lengths)
print(np.sum(traj_lengths))

np.savez("multi_onehot.npz",states=obs_buf,actions=act_buf,images=None,traj_lengths=traj_lengths)