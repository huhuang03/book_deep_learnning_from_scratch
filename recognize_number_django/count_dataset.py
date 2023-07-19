import os.path

dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime', 'dataset')

count = [0] * 10

for f in os.listdir(dataset_root):
    num = int(f[f.rindex('_')+1 : f.rindex('.jpg')])
    count[num] += 1

print(count)

