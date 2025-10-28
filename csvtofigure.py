# 导入所需的模块
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast


analysis_batch = 10000
# 读取CSV文件并进行初步处理
file_path = 'C:/AI_Facade\PARL_01/examples/MARL_for_ALL/train_log_qmix/qmix_False/log.csv'    # 替换为实际的文件路径
data = pd.read_csv(file_path)

# 将字符串格式的agents rewards转换为包含多个数值的列表
def parse_agents_rewards(rewards_str):
    rewards = ast.literal_eval(rewards_str)
    return rewards

# 将Agents Rewards列分解为多个单独的列
# 假设有9个智能体
num_agents = 7
reward_columns = [f'Agent{i+1} Reward' for i in range(num_agents)]
data[reward_columns] = pd.DataFrame(data['Agents Rewards'].apply(parse_agents_rewards).tolist(), index=data.index)

# 计算每1000个episode的平均值
episode_groups = data.groupby(data['Episode'] // analysis_batch)
mean_rewards = episode_groups['Reward'].mean()
std_rewards = episode_groups['Reward'].std()
mean_agent_rewards = episode_groups[reward_columns].mean()
std_agent_rewards = episode_groups[reward_columns].std()

# 删除不足1000个episode的最后一个数据组
filtered_mean_rewards = mean_rewards[mean_rewards.index != mean_rewards.index.max()]
filtered_std_rewards = std_rewards[std_rewards.index != std_rewards.index.max()]
filtered_mean_agent_rewards = mean_agent_rewards[mean_agent_rewards.index != mean_agent_rewards.index.max()]
filtered_std_agent_rewards = std_agent_rewards[std_agent_rewards.index != std_agent_rewards.index.max()]

# 绘制图表并保存为SVG文件
plt.figure(figsize=(10, 6))  # 设置为3:5的比例

# 总奖励
plt.plot(filtered_mean_rewards.index, filtered_mean_rewards.values, color='blue', label='Mean Total Reward')
plt.fill_between(filtered_mean_rewards.index, filtered_mean_rewards.values - filtered_std_rewards.values / 2, 
                 filtered_mean_rewards.values + filtered_std_rewards.values / 4, color='blue', alpha=0.3)

# 所有智能体的奖励及误差带
for i in range(num_agents):
    color = plt.cm.viridis(i / num_agents)  # 使用不同的颜色
    label = f'Mean Agent{i+1} Reward'
    plt.plot(filtered_mean_agent_rewards.index, filtered_mean_agent_rewards[f'Agent{i+1} Reward'].values, color=color, label=label)
    plt.fill_between(filtered_mean_agent_rewards.index, 
                     filtered_mean_agent_rewards[f'Agent{i+1} Reward'].values - filtered_std_agent_rewards[f'Agent{i+1} Reward'].values / 4, 
                     filtered_mean_agent_rewards[f'Agent{i+1} Reward'].values + filtered_std_agent_rewards[f'Agent{i+1} Reward'].values / 4, 
                     color=color, alpha=0.3)

# 设置X轴刻度
filtered_x_ticks = np.arange(0, filtered_mean_rewards.index.max() + 1, 10)  # 每隔5个显示一个刻度
plt.xticks(ticks=filtered_x_ticks, labels=[str(x) for x in filtered_x_ticks], fontsize=14)
plt.xlim(left=0, right=filtered_mean_rewards.index.max())

# 设置标题和轴标签的大小
plt.title('Average Rewards with Variance per ' + str(analysis_batch) + ' Episodes', fontsize=16)
plt.xlabel('Episode Group (x' + str(analysis_batch) + ')', fontsize=14)
plt.ylabel('Average Reward', fontsize=14)

# 设置图例的大小
plt.legend(frameon=True, fontsize=12)

# 设置全框并调整刻度线和框的粗细
spine_width = 2
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['top'].set_linewidth(spine_width)
plt.gca().spines['right'].set_linewidth(spine_width)
plt.gca().spines['bottom'].set_linewidth(spine_width)
plt.gca().spines['left'].set_linewidth(spine_width)
plt.tick_params(direction='out', length=6, width=spine_width, colors='black', labelsize=14)

# 保存为SVG文件
svg_file_path = 'C:/AI_Facade\PARL_01/examples/MARL_for_ALL/train_log_qmix/qmix_False/reward_plot.svg'  # 指定保存位置
plt.savefig(svg_file_path, format='svg')

plt.show()

svg_file_path  # 返回文件路径以供下载
