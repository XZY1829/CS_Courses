import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

epochs, train_loss, val_loss, val_bleu, lr = [], [], [], [], []
with open('runs_transformer_cuda/train_20260609_151606_v3_char_rdrop/train_log.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        epochs.append(int(row['epoch']))
        train_loss.append(float(row['train_loss']))
        val_loss.append(float(row['val_loss']))
        val_bleu.append(float(row['val_bleu']))
        lr.append(float(row['lr']))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax1 = axes[0]
ax1.plot(epochs, train_loss, 'b-', linewidth=1.5, label='Train Loss')
ax1.plot(epochs, val_loss, 'r-', linewidth=1.5, label='Val Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('训练与验证损失', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(epochs, val_bleu, 'g-', linewidth=1.5, marker='', label='Val BLEU')
ax2.axhline(y=40.89, color='orange', linestyle='--', linewidth=1, label='Test BLEU = 40.89')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('BLEU', fontsize=12)
ax2.set_title('验证集 BLEU 曲线', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
gap = [t - v for t, v in zip(val_loss, train_loss)]
ax3.plot(epochs, gap, 'm-', linewidth=1.5)
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Val Loss - Train Loss', fontsize=12)
ax3.set_title('过拟合 Gap 变化', fontsize=13)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/training_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/training_curves.png', dpi=150, bbox_inches='tight')
print('Saved: figures/training_curves.pdf & .png')
