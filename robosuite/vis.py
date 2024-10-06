import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# k 표기법 포매터 함수
def k_formatter(x, pos):
    if x >= 1000:
        return f'{x / 1000:.0f}k'
    return f'{int(x)}'

setting = 'blind'

# CSV 파일 경로
csv_file_path_list = [
    f'./dat/{setting}_0_tb.csv',
    f'./dat/{setting}_1_tb.csv',
    f'./dat/{setting}_2_tb.csv',
    f'./dat/{setting}_3_tb.csv',
    f'./dat/{setting}_4_tb.csv',
    f'./dat/{setting}_5_tb.csv',
    f'./dat/{setting}_6_tb.csv',
    f'./dat/{setting}_7_tb.csv',
    f'./dat/{setting}_8_tb.csv',
    f'./dat/{setting}_9_tb.csv',
]

# 시각화
plt.figure(figsize=(5, 5))
for csv_file_path in csv_file_path_list:
    data = pd.read_csv(csv_file_path)
    plt.plot(data['Step'], data['Value'])
if setting == 'candidate':
    plt.title(f'Episode Reward Mean: w/ Arti-info')
elif setting == 'blind':
    plt.title(f'Episode Reward Mean: w/o Arti-info')
plt.xlabel('Total Time Step')
plt.ylabel('Episode Reward')
#plt.grid()
#plt.legend()
if setting == 'candidate':
    plt.ylim([0, 400])
elif setting == 'blind':
    plt.ylim([0, 10])

plt.gca().xaxis.set_major_formatter(FuncFormatter(k_formatter))
plt.savefig(f'{setting}.png')