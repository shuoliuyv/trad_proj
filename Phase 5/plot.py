import csv
import matplotlib.pyplot as plt

order_sizes = []
base_times = []
opt_times = []

with open("benchmark_results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        order_sizes.append(int(row["orders"]))
        base_times.append(float(row["base_seconds"]))
        opt_times.append(float(row["optimized_seconds"]))

plt.figure(figsize=(10, 6))

plt.plot(order_sizes, base_times, marker="o", linestyle="-", label="Base Implementation")
plt.plot(order_sizes, opt_times, marker="s", linestyle="-", label="Optimized Implementation")

plt.xlabel("Number of Orders")
plt.ylabel("Execution Time (seconds)")
plt.title("HFT Order Book Performance Scalability")
plt.legend()
plt.grid(True)
plt.ticklabel_format(style="plain", axis="x")

plt.savefig("execution_time_chart.png", dpi=200, bbox_inches="tight")
plt.show()