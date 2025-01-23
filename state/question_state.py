# DB
from typing import TypedDict, Annotated

class GraphState(TypedDict):
    job: Annotated[str, "직무"]
    company_id: Annotated[int, '회사 식별자']
    applicant_id: Annotated[int, '지원자 식별자']
    query_main: Annotated[str, "맨 처음 들어가는 query"]
    resume: Annotated[str, "지원자 자소서"]
    evaluation: Annotated[str, "회사기준 DB"]
    relevance: Annotated[str, '관련성 체크(Yes or No)']
    final_question: Annotated[str, "생성된 질문"]