"""Generate synthetic fixtures only; never leaderboard data."""
import argparse
import csv
from pathlib import Path
from core import FIELDS

def generate(destination):
    root = Path(destination)
    for split in ('train', 'test'):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        sources = {1: [], 2: [], 3: []}
        labels = []
        names = ['Orchid', 'Maple', 'Falcon', 'Cedar', 'Lotus', 'Amber', 'River', 'Coral', 'Silver', 'Olive', 'Banyan', 'Pearl']
        for i, name in enumerate(names):
            country = 'France' if split == 'test' else ('India' if i % 2 else 'US')
            address = f'{10+i} {name} Road'
            sources[1].append([f'S1-{i:03}', f'{name} Books', address, country])
            matches = []
            for s in (2, 3):
                if i % 4 != 0:
                    target = f'S{s}-{i:03}'
                    sources[s].append([target, f'{name} Books' if s == 2 else f'{name} books ltd', address if s == 2 else address.replace('Road', 'Rd'), country])
                    matches.append(target)
                else:
                    sources[s].append([f'S{s}-{i:03}', f'Unrelated {i} Bakery', f'{900+i} Market Lane', country])
            labels.append([f'S1-{i:03}', ','.join(matches)])
        for s, rows in sources.items():
            with open(folder/f'{split}_source{s}.tsv', 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                writer.writerow(FIELDS)
                writer.writerows(rows)
        if split == 'train':
            with open(folder/'train_ground_truth.tsv', 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                writer.writerow(['source1_entity_id', 'matched_entity_ids'])
                writer.writerows(labels)
    print(f'Synthetic demo generated: {root}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='demo_data')
    generate(p.parse_args().output)
