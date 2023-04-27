import matplotlib.pyplot as plt
import numpy as np


cat_loss = np.load('targets_num_1_restart_4_attack_350/selected_subset_cat_1000_train_loss_trial0.npy')
truck_loss = np.load('targets_num_1_restart_4_attack_350/selected_subset_truck_1000_train_loss_trial0.npy')



plt.figure()
plt.plot(cat_loss)
plt.xlabel('iteration')
plt.ylabel('poison train loss')

plt.title('selected_subset_cat_1000_train_loss_trial0')
plt.savefig('targets_num_1_restart_4_attack_350/selected_subset_cat_1000_train_loss_trial0.png')

plt.figure()
plt.plot(truck_loss)
plt.xlabel('iteration')
plt.ylabel('poison train loss')

plt.title('selected_subset_truck_1000_train_loss_trial0')
plt.savefig('targets_num_1_restart_4_attack_350/selected_subset_truck_1000_train_loss_trial0.png')