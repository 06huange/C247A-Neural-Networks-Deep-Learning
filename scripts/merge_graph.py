from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# Paths to the two runs
run1_log = "logs/final/lightning_logs/version_0"
run2_log = "logs/2026-03-12/05-50-39/lightning_logs/version_0"

def load_metric(log_dir, tag="val/CER"):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)

    steps = [e.step for e in events]
    values = [e.value for e in events]

    return steps, values

# Load both runs
steps1, cer1 = load_metric(run1_log)
steps2, cer2 = load_metric(run2_log)

# Plot
plt.figure(figsize=(8,5))

plt.plot(steps1, cer1, label="Conformer(Gaussian 0.005)")
plt.plot(steps2, cer2, label="TDS")

plt.xlabel("Training Step")
plt.ylabel("Validation CER")
plt.title("Validation CER Comparison")
plt.legend()
plt.grid(True)
plt.xlim(0, 8000)
plt.ylim(0, 70)

plt.tight_layout()
plt.savefig("val_cer_comparison6.png", dpi=300)
plt.show()