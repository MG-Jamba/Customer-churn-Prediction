import matplotlib.pyplot as plt

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

precision = [0.24, 0.28, 0.32, 0.38, 0.46]
recall = [0.92, 0.83, 0.70, 0.54, 0.34]

plt.plot(
    thresholds,
    precision,
    marker="o",
    label="Churn Precision"
)

plt.plot(
    thresholds,
    recall,
    marker="o",
    label="Churn Recall"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Customer Churn: Precision vs Recall")
plt.xticks(thresholds)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("threshold_comparison.png", dpi=300)
plt.show()