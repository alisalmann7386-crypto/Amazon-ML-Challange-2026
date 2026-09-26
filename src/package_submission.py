"""Validate outputs and build the challenge's exact submission directory layout."""
import argparse
import re
import zipfile
from pathlib import Path
from core import read_sources, validate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--team', required=True)
    p.add_argument('--test-dir', help='Required for original TF-IDF baseline')
    p.add_argument('--index', help='Test SQLite index for scalable baseline')
    p.add_argument('--model', help='Scalable trained model artifact to include')
    p.add_argument('--model-dir', help='Hybrid model artifact directory')
    p.add_argument('--hybrid-index', help='Hybrid test catalog directory')
    p.add_argument('--output', default='output')
    p.add_argument('--destination', default='student_resource/Submissions')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.team):
        p.error('Use letters, digits, underscores or hyphens for team name')
    root = Path(__file__).resolve().parents[1]
    if args.hybrid_index:
        if not args.model_dir: p.error('--hybrid-index requires --model-dir')
        from validate_submission import validate as validate_hybrid
        validate_hybrid(args.hybrid_index,args.output)
    elif args.index:
        if not args.model: p.error('--index requires --model')
        from scalable import validate_index
        validate_index(args.index,args.output)
    else:
        if not args.test_dir: p.error('Provide --test-dir or --index with --model')
        validate(args.output, *read_sources(args.test_dir, 'test'))
    destination = Path(args.output) if args.hybrid_index else Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / ('submission.zip' if args.hybrid_index else f'{args.team}_submission.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            z.write(Path(args.output)/name, f'output/{name}')
        for path in sorted((root/'src').glob('*.py')):
            z.write(path, f'code/business_entity_resolution/src/{path.name}')
        if args.hybrid_index:
            z.write(root/'REPRODUCE_HYBRID.md','code/business_entity_resolution/README.md')
            for pth in sorted(Path(args.model_dir).glob('*')):
                if pth.is_file(): z.write(pth, 'code/business_entity_resolution/artifacts/final_model/'+pth.name)
            for pth in sorted((root/'configs').glob('*.json')):
                z.write(pth,'code/business_entity_resolution/configs/'+pth.name)
        elif args.index:
            z.write(root/'COLAB.md', 'code/business_entity_resolution/README.md')
            z.write(args.model, 'code/business_entity_resolution/artifacts/model.joblib')
        else:
            z.write(root/'REPRODUCE.md', 'code/business_entity_resolution/README.md')
        z.write(root/'LICENSE', 'code/business_entity_resolution/LICENSE')
        z.write(root/'requirements.txt', 'code/business_entity_resolution/requirements.txt')
        z.write(root/'Documentation_template.md', 'Documentation_template.md')
    print(f'Created {archive}; review methodology and real-data results before submission.')
