import matplotlib.pyplot as plt
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


print("Column 1:", x1)
print("Column 2:", x2)
print("Column 3:", x3)
print("Column y:", y)
print("Column 1:", normalx1)
print("Column 2:", normalx2)
print("Column 3:", normalx3)
print("Column y:", normaly)

phi0 = 0
phi1 = 0
phi2 = 0
phi3 = 0
learing_rate = 0.02


for i in range(100000):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum0 = 0
    costsum = 0
    for j in range(50):
        hyp = (phi0 + phi1 * normalx1[j] + phi2 * normalx2[j] + phi3 * normalx3[j])
        sum0 = sum0 + (hyp - normaly[j])
        sum1 = sum1 + (hyp - normaly[j]) * normalx1[j]
        sum2 = sum2 + (hyp - normaly[j]) * normalx2[j]
        sum3 = sum3 + (hyp - normaly[j]) * normalx3[j]

        costsum = costsum + (hyp - normaly[j])**2


    cost = (1/(2*50)) * costsum
    phi0 -= learing_rate * (1/50) * sum0
    phi1 -= learing_rate * (1/50) * sum1
    phi2 -= learing_rate * (1/50) * sum2
    phi3 -= learing_rate * (1/50) * sum3
    print("Iteration " , i)
    print("phi0= ", phi0)
    print("phi1= ", phi1)
    print("phi2= ", phi2)
    print("phi3= ", phi3)
    print("Cost= ", cost)


plt.scatter(normalx1, normaly, label='Data X1')
plt.scatter(normalx2, normaly, label='Data X2')
plt.scatter(normalx3, normaly, label='Data X3')
regression_line = [phi0 + phi1 * x + phi2 * y + phi3 * z for x, y, z in zip(normalx1, normalx2, normalx3)]
plt.plot(normalx1, regression_line, label='Regression Line', color='red')
plt.xlabel('Normalized X1, X2, X3')
plt.ylabel('Normalized Y')
plt.legend()
plt.title('Linear Regression')
plt.show()