import numpy as np
file_path = "DataX.dat"
file_path2 = "DataY.dat"

# Initialize empty lists to store the data for each column
x1 = []
x2 = []
x3 = []
y = []
normalx1 = []
normalx2 = []
normalx3 = []
normaly = []

with open(file_path2, 'r') as file:
    for line in file:
        # Assuming each line contains a single data point
        data_point = float(line.strip())  # Convert the line to a float
        y.append(data_point)

with open(file_path, 'r') as file:
    for line in file:
        values = line.strip().split()
        x1.append(float(values[0]))
        x2.append(float(values[1]))
        x3.append(float(values[2]))

maxvalx1 = max(x1)
minvalx1 = min(x1)
meanx1 = sum(x1)/50
for x in x1:
    normalx1.append(float((x - meanx1)/(maxvalx1 - minvalx1)))
maxvalx2 = max(x2)
minvalx2 = min(x2)
meanx2 = sum(x2)/50
for x in x2:
    normalx2.append(float((x - meanx2)/(maxvalx2 - minvalx2)))
maxvalx3 = max(x3)
minvalx3 = min(x3)
meanx3 = sum(x3)/50
for x in x3:
    normalx3.append(float((x - meanx3)/(maxvalx3 - minvalx3)))
maxvaly = max(y)
minvaly = min(y)
meany = sum(y)/50
for x in y:
    normaly.append(float((x - meany)/(maxvaly - minvaly)))
num_rows = 50
num_columns = 4
# Initialize an empty matrix with zeros
MatrixA = [[0 for _ in range(num_columns)] for _ in range(num_rows)]

# Initialize an empty matrix with zeros
MatrixY = [[0 for _ in range(1)] for _ in range(50)]

for i in range(50):
    MatrixA[i][0] = 1
    MatrixA[i][1] = normalx1[i]
    MatrixA[i][2] = normalx2[i]
    MatrixA[i][3] = normalx3[i]

for i in range(50):
    MatrixY[i][0] = normaly[i]

mat = np.array(MatrixA)
mty = np.array(MatrixY)
print("Matrix A:",mat)
print("Matrix Y:",mty)

mt = mat.T

res = np.dot(mt,mat)

pseudo_inverse = np.linalg.inv(res.T @ res) @ res.T

XTY = np.dot(mt,mty)

totalres = np.dot(pseudo_inverse,XTY)


y_predicted = np.dot(mat, totalres)
squared_errors = (mty - y_predicted) ** 2
mse = (1 / (2 * 50)) * np.sum(squared_errors)



print("Thethas: ", totalres)
print(mse)


import matplotlib.pyplot as plt

# ... Your existing code ...

# After your linear regression loop, you can add the following code to plot the data and the regression line:

# Create a scatter plot for the original data points
plt.scatter(normalx1, normaly, label='Data X1')
plt.scatter(normalx2, normaly, label='Data X2')
plt.scatter(normalx3, normaly, label='Data X3')

# Calculate the regression line based on the learned coefficients
regression_line = [totalres[0] + totalres[1] * x + totalres[2] * y + totalres[3] * z]

# Plot the regression line
plt.plot(normalx1, regression_line, label='Regression Line', color='red')

plt.xlabel('Normalized X1, X2, X3')
plt.ylabel('Normalized Y')
plt.legend()
plt.title('Closed Form Solution')

# Show the plot
plt.show()