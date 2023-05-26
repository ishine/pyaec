import numpy as np
import matplotlib.pyplot as plt

# 生成两个信号
signal_1 = np.sin(np.linspace(0, 20*np.pi, 1000))
signal_2 = np.sin(np.linspace(np.pi/4, 20*np.pi + np.pi/4, 1000))

# 计算它们之间的互相关
corr = np.correlate(signal_1, signal_2, mode='full')
delay = corr.argmax() - (len(signal_1) - 1)
print("Delay: ", delay)

# 绘制信号和互相关结果
fig, axs = plt.subplots(3, 1)

axs[0].plot(signal_1)
axs[0].set_title('Signal 1')

axs[1].plot(signal_2)
axs[1].set_title('Signal 2')

axs[2].plot(corr)
axs[2].axvline(delay + len(signal_1) - 1, color='r', linestyle='--')
axs[2].set_title('Cross-correlated Signal 1 and Signal 2')
plt.show()
