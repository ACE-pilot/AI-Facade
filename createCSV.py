import re
import csv

#tensorboard --logdir=train_log\Park_Env_True

log_file_path = 'C:/AI_Facade\PARL_01/examples/MARL_for_ALL/train_log_qmix\qmix_False/log.log'  # 替换为你的日志文件路径
csv_file_path = 'C:/AI_Facade\PARL_01/examples/MARL_for_ALL/train_log_qmix\qmix_False/log.csv'   # 替换为你想要保存的CSV文件路径

# 解析日志文件
data = []
with open(log_file_path, 'r') as file:
    for line in file:
        if 'total_steps' in line and 'Evaluation over' not in line:
            # 使用正则表达式来解析行
            match = re.search(r'total_steps (\d+), episode (\d+), reward ([-\d.]+), agents rewards \[([-\d., ]+)\], episode steps (\d+)', line)
            if match:
                total_steps = match.group(1)
                episode = match.group(2)
                reward = match.group(3)
                agents_rewards = match.group(4)
                episode_steps = match.group(5)
                data.append([total_steps, episode, reward, agents_rewards, episode_steps])

# 保存到CSV
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Total Steps', 'Episode', 'Reward', 'Agents Rewards', 'Episode Steps'])  # 写入标题
    writer.writerows(data)

print('Log data has been written to', csv_file_path)
