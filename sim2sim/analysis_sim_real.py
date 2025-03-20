import matplotlib.pyplot as plt
import numpy as np

# raisim
raisim_obs_path = 'BJ_Raisim/net/HSW/data/actor_state.csv'
raisim_obs = np.loadtxt(raisim_obs_path, delimiter=",")
# real
real_obs_path = 'deploy/data/robot_data.csv'
real_data = np.loadtxt(real_obs_path, delimiter=',')
real_obs = real_data[:, :42]

nb_rows = 6
nb_cols = 7
fig, axs = plt.subplots(nb_rows, nb_cols)
start = 0
end = 1200

for i in range(6):
    for j in range(7):
        a = axs[i,j]
        a.grid(True)
        a.plot(raisim_obs[start:end, (nb_rows+1)*i+j], c='r')
        a.plot(real_obs[start:end, (nb_rows+1)*i+j], c='g')
        a.set(title=f'OBS_{(nb_rows+1)*i+j}')

plt.show()
