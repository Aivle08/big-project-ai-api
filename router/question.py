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
# Node
from node.question_node import input, retrieve_document, relevance_check, combine_prompt, fact_checking, rewrite_question, experience_work_fact_checking
# etc
from etc.etcc import question_is_relevant, question_is_fact
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
from prompt.question_prompt import tecnology_prompt, rewrite_prompt, experience_prompt, work_prompt
# TechDTO
from dto.question_dto import TechDTO, Experience_WorkDTO
#########################################################################################
question = APIRouter(prefix='/question')

# 기술 중심 Prompt
@question.post("/tech", status_code = status.HTTP_200_OK, tags=['question'])
async def tech_langgraph(item: TechDTO):
    print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
    try:
        workflow = StateGraph(QuestionState)

        # 1. Node 추가
        workflow.add_node("input", input)
        ## Retriever
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "retrieve_2_document",
            lambda state: {"evaluation": retrieve_document(state, "evaluation", 'company_id')},
        )
        ## Relevance
        workflow.add_node(
            "relevance_check_1",
            lambda state: {"relevance_1": relevance_check(state, 'resume')},
        )
        workflow.add_node(
            "relevance_check_2",
            lambda state: {"relevance_2": relevance_check(state, 'evaluation')},
        )
        workflow.add_node(
            "rewrite_question_1",
            lambda state: {"resume_query": rewrite_question(state, rewrite_prompt, 'resume')},
        )
        workflow.add_node(
            "rewrite_question_2",
            lambda state: {"evaluation_query": rewrite_question(state, rewrite_prompt, 'evaluation')},
        )
        workflow.add_node(
            "combine_prompt",
            lambda state: {"final_question": combine_prompt(state, tecnology_prompt)},
        )
        workflow.add_node("fact_checking", fact_checking)
        # 2. Edge 연결
        workflow.add_edge("input", "retrieve_1_document")
        workflow.add_edge("input", "retrieve_2_document")
        workflow.add_edge("retrieve_1_document", "relevance_check_1")
        workflow.add_edge("retrieve_2_document", "relevance_check_2")
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check_1",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            lambda state: question_is_relevant(state, key='relevance_1'),
            {
                "relevant": "combine_prompt",  # 관련성이 있으면 답변을 생성합니다.
                "not_relevant": "rewrite_question_1",  # 관련성이 없으면 다시 검색합니다.
            },
        )
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check_2",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            lambda state: question_is_relevant(state, key = 'relevance_2'),
            {
                "relevant": "combine_prompt",  # 관련성이 있으면 답변을 생성합니다.
                "not_relevant": "rewrite_question_2",  # 관련성이 없으면 다시 검색합니다.
            },
        )
        workflow.add_edge("rewrite_question_1", "retrieve_1_document")
        workflow.add_edge("rewrite_question_2", "retrieve_2_document")
    
        workflow.add_edge('combine_prompt','fact_checking')
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "fact_checking",  # 관련성 체크 노드에서 나온 결과를 is_fact 함수에 전달합니다.
            question_is_fact,
            {
                "fact": END,  # 관련성이 있으면 답변을 생성합니다.
                "not_fact": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
            },
        )
        

        # 4. 그래프 진입점 설정
        workflow.set_entry_point("input")

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
                                )


        # 10. 그래프 실행 출력
        invoke_graph(app, inputs, config)

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
            'item': outputs["final_question"]
        }
    except RecursionError:  # 재귀 한도 초과 시 예외 처리
        print("\033[31m[재귀 한도 초과]\033[0m")
        #print(outputs.items())
        outputs = app.get_state(config).values 
        return {
            "status": "success",
            "code": 200,
            "message": "재귀 한도를 초과하여 판단 불가.",
            'item': '재귀 한도를 초과하여 판단 불가.'
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
    print('\n\033[36m[AI-API] \033[32m 질문 추출(경험)')
    try:
        workflow = StateGraph(QuestionState)

        # 1. Node 추가
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "relevance_check",
            lambda state: {"relevance_1": relevance_check(state, 'resume')},
        )
        workflow.add_node(
            "rewrite_question",
            lambda state: {"resume_query": rewrite_question(state, rewrite_prompt, 'resume')},
        )
        ## Relevance
        workflow.add_node(
            "fact_check",
            lambda state: {"fact": experience_work_fact_checking(state, 'resume')},
        )
        workflow.add_node(
            "combine_prompt",
            lambda state: {"final_question": combine_prompt(state, experience_prompt)},
        )

        # 2. Edge 추가
        workflow.add_edge('retrieve_1_document','relevance_check')
        workflow.add_edge('combine_prompt','fact_check')
        workflow.add_edge('rewrite_question','retrieve_1_document')
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            lambda state: question_is_relevant(state, 'relevance_1'),
            {
                "relevant": 'combine_prompt',  # 관련성이 있으면 답변을 생성합니다.
                "not_relevant": "rewrite_question",  # 관련성이 없으면 다시 검색합니다.
            },
        )
        
        workflow.add_conditional_edges(
            "fact_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            lambda state: question_is_fact(state),
            {
                "fact": END,  # 관련성이 있으면 답변을 생성합니다.
                "not_fact": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
            },
        )

        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_1_document")

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
                            resume_query=f'{item.job}의 기술 중심으로 생성해줘' )


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
            'item': outputs["final_question"]
        }
    except RecursionError:  # 재귀 한도 초과 시 예외 처리
        print("\033[31m[재귀 한도 초과]\033[0m")
        outputs = app.get_state(config).values 
        return {
            "status": "success",
            "code": 200,
            "message": "재귀 한도를 초과하여 판단 불가.",
            'item': '재귀 한도를 초과하여 판단 불가.'
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
    print('\n\033[36m[AI-API] \033[32m 질문 추출(경력)')
    try:
        workflow = StateGraph(QuestionState)

        # 1. Node 추가
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "relevance_check",
            lambda state: {"relevance_1": experience_work_fact_checking(state, 'resume')},
        )
        workflow.add_node(
            "rewrite_question",
            lambda state: {"resume_query": rewrite_question(state, rewrite_prompt, 'resume')},
        )
        ## Relevance
        workflow.add_node(
            "fact_check",
            lambda state: {"fact": experience_work_fact_checking(state, 'resume')},
        )
        
        workflow.add_node(
            "combine_prompt",
            lambda state: {"final_question": combine_prompt(state, work_prompt)},
        )

        # 2. Edge 추가
        workflow.add_edge('retrieve_1_document','relevance_check')
        workflow.add_edge('rewrite_question','retrieve_1_document')
        workflow.add_edge('combine_prompt','fact_check')
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            lambda state: question_is_relevant(state, 'relevance_1'),
            {
                "relevant": "combine_prompt",  # 관련성이 있으면 답변을 생성합니다.
                "not_relevant": "rewrite_question",  # 관련성이 없으면 다시 검색합니다.
            },
        )
        
        workflow.add_conditional_edges(
            "fact_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            lambda state: question_is_fact(state),
            {
                "fact": END,  # 관련성이 있으면 답변을 생성합니다.
                "not_fact": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
            },
        )


        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_1_document")

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
                            resume_query=f'경력 사항, 인턴 및 대외활동')

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
            'item': outputs["final_question"]
        }
    except RecursionError:  # 재귀 한도 초과 시 예외 처리
        print("\033[31m[재귀 한도 초과]\033[0m")
        outputs = app.get_state(config).values 
        return {
            "status": "success",
            "code": 200,
            "message": "재귀 한도를 초과하여 판단 불가.",
            'item': '재귀 한도를 초과하여 판단 불가.'
        }
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }