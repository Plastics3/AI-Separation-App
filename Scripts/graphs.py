import json
import matplotlib.pyplot as plt

path = r'input here'  # change to your json file path


with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(data['train_loss_history'], label='train', color='g')
ax.plot(data['valid_loss_history'], label='valid', color='m')

ax.scatter(
    data['best_epoch'] - 1,
    data['best_loss'],
    s=150,
    marker='v',
    color='c',
    label='best'
)

ax.annotate(
    f"{data['best_loss']:.3f}",
    (data['best_epoch'] - 1, data['best_loss']),
    textcoords="offset points",
    xytext=(0, -25),
    ha='center',
    fontsize=18,
    color='c'
)

ax.set_xlabel('epochs', fontsize=26)
ax.set_ylabel('mse loss', fontsize=26)
ax.tick_params(axis='both', labelsize=20)
ax.legend(prop={'size': 26})
ax.set_yscale('log')


# dynamic y-limit (optional but recommended)
y_max = max(
    max(data['train_loss_history']),
    max(data['valid_loss_history'])
)
ax.set_ylim(0, y_max * 1.05)

plt.show()
