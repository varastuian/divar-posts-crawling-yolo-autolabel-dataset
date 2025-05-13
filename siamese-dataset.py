import os
import random
import csv

fronts = os.listdir('Siamese data/front')
rears  = os.listdir('Siamese data/rear')

pairs = []
# Positive front–front
for _ in range(1000):
    a, b = random.sample(fronts, 2)
    pairs.append(('Siamese data/front/'+a, 'Siamese data/front/'+b, 1))
# Positive rear–rear
for _ in range(1000):
    a, b = random.sample(rears, 2)
    pairs.append(('Siamese data/rear/'+a, 'Siamese data/rear/'+b, 1))
# Negative front–rear
for _ in range(2000):
    a = random.choice(fronts)
    b = random.choice(rears)
    pairs.append(('Siamese data/front/'+a, 'Siamese data/rear/'+b, 0))

# Shuffle and save
random.shuffle(pairs)
with open('Siamese data/pairs2.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['img1','img2','label'])
    writer.writerows(pairs)
