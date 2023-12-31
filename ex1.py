import numpy as np
import matplotlib.pyplot as plt

money_parameters = (1800 , 200)
height_parameters = (170, 10)

# Step 1
def distribution(info):
    mean , std = info
    return np.random.normal(mean, std, 1000)

x_sample = distribution(money_parameters)
y_sample = distribution(height_parameters)
sample_x_mean = np.mean(x_sample)
sample_y_mean = np.mean(y_sample)


print(f"1)\nSince i used a normal distribution, for both height and money,\nthe expected value for Z = (X,Y)\nis the mean of the distributions :\nZ = ({sample_x_mean}, {sample_y_mean})")
print(f"Expected Value for Money (X): {sample_x_mean}")
print(f"Expected Value for Height (Y): {sample_y_mean}")

# Step 2
plt.scatter(x_sample, y_sample, alpha=0.5)
plt.xlabel('Money (X)')
plt.ylabel('Height (Y)')
plt.show()

# Step 3
euclidean_distances = []
sample_sizes = []

# Step 3: Compute empirical average and Euclidean distance for increasing values of n
for n in range(1,500):
    x_subset = np.random.choice(x_sample , size=n, replace=False) 
    y_subset = np.random.choice(y_sample , size=n, replace=False) 
    empirical_average = (np.mean(x_subset), np.mean(y_subset))
    euclidean_distance = np.linalg.norm(empirical_average - np.array([money_parameters[0], height_parameters[0]]))
    # Append to lists
    euclidean_distances.append(euclidean_distance)
    sample_sizes.append(n)

# Step 4: Plot Euclidean Distance vs. Number of Samples
plt.plot(sample_sizes, euclidean_distances)
plt.title('Euclidean Distance by Number of Samples')
plt.xlabel('Number of Samples')
plt.ylabel('Euclidean Distance')
plt.show()