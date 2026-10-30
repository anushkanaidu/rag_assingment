"""Run retrieval metrics and optional same-model RAG vs LLM comparison.
Reference judgments are document-level, not exhaustive chunk-level; see README.
"""
import csv, argparse, json
from pathlib import Path
from rag import RagEngine,answer

def run(compare=False):
    engine=RagEngine();engine.ingest()
    rows=list(csv.DictReader(open('tests/questions.csv',encoding='utf-8')))
    output=[]
    for row in rows:
        hits=engine.retrieve(row['question'],k=3)
        retrieved=[x['source'] for x in hits]
        gold=set(filter(None,row['relevant_sources'].split(';')))
        # Unique document sources: doc-level P@3, R@3; unanswerable queries are excluded.
        unique=list(dict.fromkeys(retrieved))
        matching=len(gold.intersection(unique))
        result={**row,'retrieved_top3':';'.join(retrieved),
                'precision_at_3':round(matching/len(unique),3) if gold else '',
                'recall_at_3':round(matching/len(gold),3) if gold else '',
                'unanswerable':not bool(gold)}
        if compare:
            result['rag_answer']=answer(row['question'],hits)
            result['pure_llm_answer']=answer(row['question'],None)
        output.append(result)
    Path('results').mkdir(exist_ok=True)
    with open('results/evaluation.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    applicable=[x for x in output if not x['unanswerable']]
    means={m:round(sum(float(x[m]) for x in applicable)/len(applicable),3) for m in ('precision_at_3','recall_at_3')}
    print(json.dumps({'n_questions':len(output),'n_scored':len(applicable),**means,'comparison_generated':compare},indent=2))
    print('Saved results/evaluation.csv')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--compare',action='store_true',help='Calls Groq twice for each question; requires API key')
    run(parser.parse_args().compare)
