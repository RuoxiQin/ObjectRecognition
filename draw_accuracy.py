import re
import matplotlib.pyplot as plt


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

with open("accuracy_record.txt") as f:
    lines = [line.rstrip('\n') for line in f if line[0] == "{"]
data = []
for line in lines:
    data.append([float(s) for s in re.split(" |,|\}", line) if isfloat(s)])
accuracy = [record[0] for record in data]
train_step = [record[2] for record in data]
plt.plot(train_step, accuracy)
plt.title("Learning Curve")
plt.xlabel("Training Step")
plt.ylabel("Accuracy")
plt.show()
