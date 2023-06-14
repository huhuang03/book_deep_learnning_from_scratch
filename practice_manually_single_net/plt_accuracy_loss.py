from matplotlib import pyplot as plt
from mongo_util import get_collection

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

conn = get_collection('manually_single_layer')
data = conn.aggregate([
    {'$sample': {'size': 1000}},
    {'$project': {'index': True, 'accuracy': True, 'loss': True}},
    {'$sort': {'index': 1}}
])
items = list(data)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('次数')
ax1.set_ylabel('准确率accuracy')
ax1.plot([x['index'] for x in items], [x['accuracy'] for x in items], color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.set_ylabel('损失loss')
ax2.plot([x['index'] for x in items], [x['loss'] for x in items], color=color)
ax1.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
#
# plt.plot([x['index'] for x in items], [x['accuracy'] for x in items])
# plt.show()
