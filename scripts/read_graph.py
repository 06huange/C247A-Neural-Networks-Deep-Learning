from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

log_dir = "logs/final/lightning_logs/version_0"

# Load TensorBoard logs
ea = EventAccumulator(log_dir)
ea.Reload()

# Extract val/CER values
events = ea.Scalars("val/CER")

steps = [e.step for e in events]
values = [e.value for e in events]

# Plot
plt.figure(figsize=(8,5))
plt.plot(steps, values)

plt.xlabel("Training Step")
plt.ylabel("Character Error Rate (CER)")
plt.title("Validation CER During Training")

plt.grid(True)
plt.show()

plt.tight_layout()
plt.savefig("val_cer_plot.png", dpi=300)