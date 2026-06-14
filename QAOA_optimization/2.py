import re
import matplotlib.pyplot as plt

iterations = []
losses = []

with open("loss.txt", "r", encoding="utf-8") as f:
    for line in f:
        m = re.search(
            r"Iteration\s+(\d+)\s+\|\s+Current Loss:\s+([-+]?\d*\.\d+|[-+]?\d+)",
            line
        )
        if m:
            iterations.append(int(m.group(1)))
            losses.append(float(m.group(2)))

plt.figure(figsize=(10,5))
plt.plot(iterations, losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("QAOA Training Loss")
plt.grid(True)
plt.show()