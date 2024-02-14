import numpy as np
import math
file_path = "DataX.dat"
file_path2 = "ClassY.dat"

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

phi0 = 0
phi1 = 0
phi2 = 0
phi3 = 0
learing_rate = 0.02
convergence_threshold = 0.0001
prev_cost = float('inf')

for i in range(10000):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum0 = 0
    cost = 0
    for j in range(50):
        hyp = phi0 + phi1 * normalx1[j] + phi2 * normalx2[j] + phi3 * normalx3[j] 
        hypprob = 1 / (1 + math.exp(-hyp))
        sum0 += (hypprob - normaly[j])
        sum1 += (hypprob - normaly[j]) * normalx1[j]
        sum2 += (hypprob - normaly[j]) * normalx2[j]
        sum3 += (hypprob - normaly[j]) * normalx3[j]
        cost += normaly[j] * math.log(hypprob) + (1 - normaly[j]) * math.log(1 - hypprob)

    cost = -cost / 50
    phi0 -= learing_rate * (1/50) * sum0
    phi1 -= learing_rate * (1/50) * sum1
    phi2 -= learing_rate * (1/50) * sum2
    phi3 -= learing_rate * (1/50) * sum3

    if abs(prev_cost - cost) < convergence_threshold:
        print("Converged at iteration", i)
        break

    prev_cost = cost


    print("Iteration " , i)
    print("phi0= ", phi0)
    print("phi1= ", phi1)
    print("phi2= ", phi2)
    print("phi3= ", phi3)
    print("Cost= ", cost)


print("Iteration " , i)
print("phi0= ", phi0)
print("phi1= ", phi1)
print("phi2= ", phi2)
print("phi3= ", phi3)
print("Cost= ", cost)