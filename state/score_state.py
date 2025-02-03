# Type
from typing import TypedDict, Annotated

class ScoreState(TypedDict):
    job: Annotated[str, "직무"]
    applicant_id: Annotated[int, '지원자 식별자']
    query_main: Annotated[str, "맨 처음 들어가는 query"]
    resume: Annotated[str, "지원자 자소서"]
    eval_item: Annotated[str, "평가 항목"]
    eval_item_content : Annotated[str, "회사 평가 기준"]
    yes_or_no: Annotated[str, '관련성 체크(Yes or No)']
    eval_resume: Annotated[str, "평가 결과"]