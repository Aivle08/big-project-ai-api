# DB
from typing import TypedDict, Annotated

class QuestionState(TypedDict):
    job: Annotated[str, "직무"]
    company_id: Annotated[int, '회사 식별자']
    applicant_id: Annotated[int, '지원자 식별자']
    resume_query: Annotated[str, "맨 처음 resume에 들어가는 query list"]
    evaluation_query: Annotated[str, "맨 처음 evaluation에 들어가는 query list"]
    resume: Annotated[str, "지원자 자소서"]
    resume_chunk: Annotated[list, '검색한 청크 리스트']
    evaluation: Annotated[str, "회사기준 DB"]
    relevance_1: Annotated[str, '관련성 체크(Yes or No)']
    relevance_2: Annotated[str, '관련성 체크(Yes or No)']
    fact: Annotated[str, '관련성 체크(Yes or No)']
    final_question: Annotated[str, "생성된 질문"]