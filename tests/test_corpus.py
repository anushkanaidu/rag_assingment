from pathlib import Path
import csv

def test_ten_independent_documents():
    files=list(Path('documents').glob('*.md'))
    assert len(files)==10
    assert all('fictional' in f.read_text().lower() for f in files)

def test_question_variety():
    questions=list(csv.DictReader(open('tests/questions.csv')))
    assert len(questions)==10
    assert len({row['type'] for row in questions})>=4
    assert sum(not row['relevant_sources'] for row in questions)>=1
