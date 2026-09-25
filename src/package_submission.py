"""Validate outputs and build the challenge's exact submission directory layout."""
import argparse
import re
import zipfile
from pathlib import Path
from core import read_sources, validate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--team', required=True)
    p.add_argument('--test-dir', required=True)
    p.add_argument('--output', default='output')
    p.add_argument('--destination', default='student_resource/Submissions')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.team):
        p.error('Use letters, digits, underscores or hyphens for team name')
    root = Path(__file__).resolve().parents[1]
    validate(args.output, *read_sources(args.test_dir, 'test'))
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f'{args.team}_submission.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            z.write(Path(args.output)/name, f'output/{name}')
        for path in sorted((root/'src').glob('*.py')):
            z.write(path, f'code/business_entity_resolution/src/{path.name}')
        z.write(root/'REPRODUCE.md', 'code/business_entity_resolution/README.md')
        z.write(root/'LICENSE', 'code/business_entity_resolution/LICENSE')
        z.write(root/'requirements.txt', 'code/business_entity_resolution/requirements.txt')
        z.write(root/'Documentation_template.md', 'Documentation_template.md')
    print(f'Created {archive}; review methodology and real-data results before submission.')
