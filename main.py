from firstdataset import *
from seconddataset import * 
import matplotlib.pyplot as plt
zvalues1 = firstdataset()
zvalues2 = seconddataset()

person = 0
ypoints = []
xpoints = []

for z1 , z2 in zip(zvalues1 , zvalues2):
    person += 1
    combinedz = 0.6 * z1 + 0.4 * z2
    xpoints.append(combinedz)
    risk = 1 / (1 + np.exp(-combinedz))
    riskpercent = round(risk * 100)
    ypoints.append(riskpercent)
    print(f"Person {person}'s risk of cancer: {riskpercent}%")


plt.xlim(min(xpoints) , max(xpoints))
ypad = 0.05 * max(ypoints)
plt.ylim(0,max(ypoints) + ypad)
plt.scatter(xpoints, ypoints, alpha=0.6)

plt.title("Cancer risk Assessment")
plt.ylabel("Predicted risk of Cancer")
plt.xlabel("overall risk score of all features (clinical + environmental + symptomatic)")
plt.show()




