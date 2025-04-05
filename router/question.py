#########################################################################################
# Basic
import os
import pandas as pd
import traceback
# Fastapi
from fastapi import APIRouter, HTTPException, status, File, UploadFile

# LangChain
from langchain_core.runnables import RunnableConfig

# Graph
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

## Module
# State
from state.question_state import QuestionState

# workflow
from workflow.question_workflow import tech_stategraph, experience_stategraph, work_stategraph
# etc
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
# TechDTO
from dto.question_dto import TechDTO, Experience_WorkDTO
#########################################################################################
question = APIRouter(prefix='/question')

# 기술 중심 Prompt
@question.post("/tech", status_code = status.HTTP_200_OK, tags=['question'])
async def tech_langgraph(item: TechDTO):
    """
    기술 중심 질문을 생성하는 LangGraph 기반 워크플로우 실행.
    - 지원자의 이력서 및 기업 평가 기준을 검색하고 관련성 검토 후 질문 생성.
    - 질문이 관련성이 낮으면 재작성 후 다시 검색을 수행.
    - Fact-checking을 거쳐 최종 질문 반환.
    """
    print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
    try:
        workflow = tech_stategraph(StateGraph(QuestionState))

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        # 7. 그래프 시각화
        visualize_graph(app,'tech_graph')
        
        # 8. config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=15, configurable={"thread_id": random_uuid()})

        # 9. 질문 입력
        inputs = QuestionState(job=item.job, 
                                company_id = item.company_id,
                                applicant_id = item.applicant_id,
                                fact='yes',
                                resume_query=f'{item.job}의 기술 중심으로 생성해줘',
                                evaluation_query=f'{item.job}의 기술 중심으로 생성해줘',
                                resume_chunk=[],
                                )


        # 10. 그래프 실행 출력
        # invoke_graph(app, inputs, config)

        # 11. 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'resume_query:\n{outputs["resume_query"]}')
        print(f'evaluation_query:\n{outputs["evaluation_query"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'evaluation:\n{outputs["evaluation"]}')
        print(f'relevance_1:\n{outputs["relevance_1"]}')
        print(f'relevance_2:\n{outputs["relevance_2"]}')
        print(f'fact:\n{outputs["fact"]}')
        print(f'final_question:\n{outputs["final_question"]}')
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "질문 생성 완료",  # 응답 메시지
            'item': {
                'question': outputs["final_question"],
                'chunk':outputs['resume_chunk'],
            }
        }
    except RecursionError:  # 재귀 한도 초과 시 예외 처리
        print("\033[31m[재귀 한도 초과]\033[0m")
        #print(outputs.items())
        outputs = app.get_state(config).values 
        return {
            "status": "success",
            "code": 204,
            "message": "재귀 한도를 초과하여 판단 불가.",
            'item': {
                'question': None,
                'chunk': None,
            }
        }
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }
            
# 경험 중심 Prompt
@question.post("/experience", status_code = status.HTTP_200_OK, tags=['question'])
def experience_langgraph(item: Experience_WorkDTO):
    """
    경험 중심 질문을 생성하는 LangGraph 기반 워크플로우 실행.
    - 지원자의 업무 경험을 기반으로 질문을 생성하고 관련성 검토 후 최적화.
    - Fact-checking을 수행하여 신뢰도를 보장.
    """
    print('\n\033[36m[AI-API] \033[32m 질문 추출(경험)')
    try:
        workflow = experience_stategraph(StateGraph(QuestionState))

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        visualize_graph(app,'experience_graph')

        # config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=15, configurable={"thread_id": random_uuid()})

        # 질문 입력
        inputs = QuestionState(job=item.job, 
                            company_id = item.company_id,
                            applicant_id = item.applicant_id,
                            evaluation = item.evaluation,
                            resume_query=f'{item.job}의 기술 중심으로 생성해줘',
                            resume_chunk=[])


        # 그래프 실행
        invoke_graph(app, inputs, config)

        # 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'resume_query:\n{outputs["resume_query"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'evaluation:\n{outputs["evaluation"]}')
        print(f'relevance_1:\n{outputs["relevance_1"]}')
        print(f'final_question:\n{outputs["final_question"]}')

        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "질문 생성 완료",  # 응답 메시지
            'item': {
                'question': outputs["final_question"],
                'chunk':outputs['resume_chunk'],
            }
        }
    except RecursionError:  # 재귀 한도 초과 시 예외 처리
        print("\033[31m[재귀 한도 초과]\033[0m")
        outputs = app.get_state(config).values 
        return {
            "status": "success",
            "code": 204,
            "message": "재귀 한도를 초과하여 판단 불가.",
            'item': {
                'question':None,
                'chunk':None,
            }
        }
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }
            
# 경력 중심 Prompt
@question.post("/work", status_code = status.HTTP_200_OK, tags=['question'])
def work_langgraph(item: Experience_WorkDTO):
    '''
    경력 중심 질문을 생성하는 LangGraph 기반 워크플로우 실행.
    - 지원자의 경력, 인턴 및 대외활동을 기반으로 질문을 생성하고 관련성을 검토.
    - 관련성이 낮으면 질문을 재작성하여 검색을 최적화.
    - Fact-checking을 수행하여 신뢰도를 보장하고 최종 질문을 출력.
    '''
    print('\n\033[36m[AI-API] \033[32m 질문 추출(경력)')
    try:
        workflow = work_stategraph(StateGraph(QuestionState))

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        visualize_graph(app,'work_graph')
            
        # config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=15, configurable={"thread_id": random_uuid()})

        # 질문 입력
        inputs = QuestionState(job=item.job, 
                            company_id = item.company_id,
                            applicant_id = item.applicant_id,
                            evaluation = item.evaluation,
                            resume_query=f'경력 사항, 인턴 및 대외활동',
                            resume_chunk=[],)

        # 그래프 실행
        invoke_graph(app, inputs, config)

        # 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'resume_query:\n{outputs["resume_query"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'evaluation:\n{outputs["evaluation"]}')
        print(f'relevance_1:\n{outputs["relevance_1"]}')
        print(f'final_question:\n{outputs["final_question"]}')
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "질문 생성 완료",  # 응답 메시지
            'item': {
                'question': outputs["final_question"],
                'chunk':outputs['resume_chunk'],
            }
        }
    except RecursionError:  # 재귀 한도 초과 시 예외 처리
        print("\033[31m[재귀 한도 초과]\033[0m")
        outputs = app.get_state(config).values 
        return {
            "status": "success",
            "code": 204,
            "message": "재귀 한도를 초과하여 판단 불가.",
            'item': {
                'question': None,
                'chunk': None,
            }
        }
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }