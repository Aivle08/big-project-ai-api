# DB
from typing import TypedDict, Annotated

class SummaryState(TypedDict):
    job: Annotated[str, "직무"]
    applicant_id: Annotated[int, '지원자 식별자']
    resume: Annotated[str, "지원자 자소서"]
    summary_result: Annotated[str, "요약 결과"]
    yes_or_no: Annotated[str, "출력 형식(Yes or No)"]
    
class yonghunState(TypedDict):
    job: Annotated[str, "직무"]
    query_main: Annotated[str, "맨 처음 들어가는 query"]
    applicant_id: Annotated[int, '지원자 식별자']
    resume: Annotated[str, "지원자 자소서"]
    output_form: Annotated[str, "출력 형식식"]
    final_result: Annotated[str, "요약 결과"]