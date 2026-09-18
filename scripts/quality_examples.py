"""Small live quality review using fictional student data; never writes class records."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import dotenv_values


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('before', 'after'):
        raise SystemExit('Usage: quality_examples.py before|after')
    key = dotenv_values(ROOT / '.env.local').get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not key:
        raise SystemExit('No configured key')
    os.environ['OPENAI_API_KEY'] = key
    from app.answer_generator import AnswerGenerator
    from app.discussion_evaluator import PersonalEvaluator
    from app.question_generator2 import QuestionGenerator2
    records = {}
    with contextlib.redirect_stdout(io.StringIO()):
        source = QuestionGenerator2()
        source.load_educational_content(str(ROOT / 'educational_content.json'))
        answer = AnswerGenerator()
        evaluator = PersonalEvaluator()
        question = '1년 뒤 300만 원이 필요한데 매달 10만 원만 모을 수 있어요. 제 계획을 어떻게 바꾸면 좋을까요?'
        records['qa'] = answer.generate_answer('가상학생', '저축 목표와 계획',
            source.get_video_script('financial_1'), source.get_slide_content_text('financial_1'), question, [])
        messages = [{'text':'1년 뒤 300만 원이 필요하고 매달 10만 원씩 모을 계획입니다.'},
                    {'text':'부족한 돈을 어떻게 맞출지 아직 정하지 못했습니다.'}]
        records['personal'] = evaluator.evaluate_user('가상학생', messages, {'topic':'저축 목표와 계획'})
        records['overall'] = evaluator.evaluate_discussion_overall([
            {'nickname':'가상학생','text':messages[0]['text']},
            {'nickname':'다른학생','text':'목표 금액을 낮추거나 기간을 늘리면 좋겠습니다.'}], {'topic':'저축 목표와 계획'})
        records['question'] = source.generate_question('가상학생', '저축 목표와 계획',
            source.get_video_script('financial_1'), source.get_slide_content_text('financial_1'),
            [{'nickname':'가상학생','text':messages[0]['text']}])
    output = ROOT / 'tmp/quality-restoration' / f'live-{sys.argv[1]}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
