from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# 데이터 모델
class GradeRetrievalQuestion(BaseModel):
    """A binary score to determine the relevance of the retrieved documents to the question."""

    score: str = Field(
        description="Whether the retrieved context is relevant to the question, 'yes' or 'no'"
    )


# 데이터 모델
class GradeRetrievalAnswer(BaseModel):
    """A binary score to determine the relevance of the retrieved documents to the answer."""

    score: str = Field(
        description="Whether the retrieved context is relevant to the answer, 'yes' or 'no'"
    )


class OpenAIRelevanceGrader:
    """
    OpenAI 기반의 관련성 평가기 클래스입니다.

    이 클래스는 검색된 문서가 주어진 질문이나 답변과 얼마나 관련이 있는지 평가합니다.
    'retrieval-question' 또는 'retrieval-answer' 두 가지 모드로 작동할 수 있습니다.

    Attributes:
        llm: 사용할 언어 모델 인스턴스
        structured_llm_grader: 구조화된 출력을 생성하는 LLM 인스턴스
        grader_prompt: 평가에 사용될 프롬프트 템플릿

    Args:
        llm: 사용할 언어 모델 인스턴스
        target (str): 평가 대상 ('retrieval-question' 또는 'retrieval-answer')
    """

    def __init__(self, llm, target="retrieval-question"):
        """
        OpenAIRelevanceGrader 클래스의 초기화 메서드입니다.

        Args:
            llm: 사용할 언어 모델 인스턴스
            target (str): 평가 대상 ('retrieval-question' 또는 'retrieval-answer')

        Raises:
            ValueError: 유효하지 않은 target 값이 제공될 경우 발생
        """
        self.llm = llm

        if target == "retrieval-question":
            self.structured_llm_grader = llm.with_structured_output(
                GradeRetrievalQuestion
            )
        elif target == "retrieval-answer":
            self.structured_llm_grader = llm.with_structured_output(
                GradeRetrievalAnswer
            )
        else:
            raise ValueError(f"Invalid target: {target}")

        # 프롬프트
        target_variable = (
            "user question" if target == "retrieval-question" else "answer"
        )
        system = f"""You are a grader assessing relevance of a retrieved document to a {target_variable}. \n 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            If the document contains keyword(s) or semantic meaning related to the {target_variable}, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to {target_variable}."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"Retrieved document: \n\n {{context}} \n\n {target_variable}: {{input}}",
                ),
            ]
        )
        self.grader_prompt = grade_prompt

    def create(self):
        """
        관련성 평가기를 생성하고 반환합니다.

        Returns:
            관련성 평가를 수행할 수 있는 체인 객체
        """

        retrieval_grader_oai = self.grader_prompt | self.structured_llm_grader
        return retrieval_grader_oai


class GroundnessQuestionScore(BaseModel):
    """Binary scores for relevance checks"""

    score: str = Field(
        description="relevant or not relevant. Answer 'yes' if the answer is relevant to the question else answer 'no'"
    )


class GroundnessAnswerRetrievalScore(BaseModel):
    """Binary scores for relevance checks"""

    score: str = Field(
        description="relevant or not relevant. Answer 'yes' if the answer is relevant to the retrieved document else answer 'no'"
    )

class GroundnessQuestionRetrievalScore(BaseModel):
    """Binary scores for relevance checks"""

    score: str = Field(
        description="relevant or not relevant. Answer 'yes' if the question is relevant to the retrieved document else answer 'no'"
    )
    
class SummaryFactCheckScore(BaseModel):
    """Binary scores for factual accuracy checks between original and summarized documents"""

    score: str = Field(
        description="Answer 'yes' if the summarized document is factually correct and accurately represents the content of the original document, otherwise answer 'no'."
    )
    
class QuestionFactCheckScore(BaseModel):
    """Binary scores for fact checks between question and retriever documents"""

    score: str = Field(
        description="fact or not fact. Answer 'Yes' if the generated question is factually correct and accurately indicates the content of the original document, else answer 'no'"
    )

class EvaluationFactCheckScore(BaseModel):
    """Binary scores for factual accuracy checks between original and summarized documents"""

    score: str = Field(
        description="Answer 'yes' if the summarized document is factually correct and accurately represents the content of the original document, otherwise answer 'no'."
    )


class GroundednessChecker:
    """
    GroundednessChecker 클래스는 문서의 정확성을 평가하는 클래스입니다.

    이 클래스는 주어진 문서가 정확한지 여부를 평가합니다.
    'yes' 또는 'no' 두 가지 중 하나를 반환합니다.

    Attributes:
        llm (BaseLLM): 사용할 언어 모델 인스턴스
        target (str): 평가 대상 ('retrieval-answer', 'question-answer' 또는 'question-retrieval')
    """

    def __init__(self, llm, target="retrieval-answer"):
        """
        GroundednessChecker 클래스의 생성자입니다.

        Args:
            llm (BaseLLM): 사용할 언어 모델 인스턴스
            target (str): 평가 대상 ('retrieval-answer', 'question-answer' 또는 'question-retrieval')
        """
        self.llm = llm
        self.target = target

    def create(self):
        """
        정확성 평가를 위한 체인을 생성합니다.

        Returns:
            Chain: 정확성 평가를 수행할 수 있는 체인 객체
        """
        # 파서
        if self.target == "retrieval-answer":
            llm = self.llm.with_structured_output(GroundnessAnswerRetrievalScore)
        elif self.target == "question-answer":
            llm = self.llm.with_structured_output(GroundnessQuestionScore)
        elif self.target == "generate-question-retrieval":
            llm = self.llm.with_structured_output(GroundnessQuestionRetrievalScore)
        elif self.target == "score-question-retrieval":
            llm = self.llm.with_structured_output(GroundnessQuestionRetrievalScore)
        elif self.target == "question-fact-check":
            llm = self.llm.with_structured_output(QuestionFactCheckScore)
        elif self.target == "score-fact-check":
            llm = self.llm.with_structured_output(SummaryFactCheckScore)
        elif self.target == "summary-question-retrieval":
            llm = self.llm.with_structured_output(GroundnessQuestionRetrievalScore)
        elif self.target == "summary-fact-check":
            llm = self.llm.with_structured_output(SummaryFactCheckScore)
        else:
            raise ValueError(f"Invalid target: {self.target}")

        # 프롬프트 선택
        if self.target == "retrieval-answer":
            # 검색된 문서와 사용자 질문의 관련성을 평가하는 채점자입니다.\n
            # 다음은 검색된 문서입니다:\n\n {context} \\n\n
            # 다음은 답입니다: {답변} \n
            # 문서에 사용자 답변과 관련된 키워드 또는 의미론적 의미가 포함된 경우 관련성이 있는 것으로 평가합니다.\n

            # 검색된 문서가 답변과 관련이 있는지 여부를 나타내기 위해 이진 점수 '예' 또는 '아니오'를 제공합니다.
            template = """You are a grader assessing relevance of a retrieved document to a user question. \n 
                Here is the retrieved document: \n\n {context} \n\n
                Here is the answer: {answer} \n
                If the document contains keyword(s) or semantic meaning related to the user answer, grade it as relevant. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the answer."""
            input_vars = ["context", "answer"]

        elif self.target == "question-answer":
            # 당신은 주어진 질문에 답이 적절하게 대응하는지 평가하는 채점자입니다.\n
            # 여기 질문이 있습니다:\n\n {질문} \n\n
            # 다음은 답입니다: {답변} \n
            # 답변이 질문을 직접 다루고 관련 정보를 제공하는 경우 관련성이 있는 것으로 평가합니다.\n
            # 평가에서 의미적 의미와 사실적 정확성을 모두 고려하세요.\n

            # 이진 점수 '예' 또는 '아니오'를 부여하여 답이 질문과 관련이 있는지 여부를 나타냅니다.
            template = """You are a grader assessing whether an answer appropriately addresses the given question. \n
                Here is the question: \n\n {question} \n\n
                Here is the answer: {answer} \n
                If the answer directly addresses the question and provides relevant information, grade it as relevant. \n
                Consider both semantic meaning and factual accuracy in your assessment. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the answer is relevant to the question."""
            input_vars = ["question", "answer"]

        elif self.target == "generate-question-retrieval":
            # 당신은 검색된 문서가 주어진 질문과 관련이 있는지 평가하는 임무를 맡은 채점자입니다.

            # ### 지침:
            # 1. 검색된 문서에 질문에 답변하거나 맥락을 제공하는 데 도움이 될 수 있는 정보가 포함되어 있는지 평가합니다.
            # 2. 문서에 질문에 직접적으로 관련되거나 간접적으로 도움이 되는 정보가 포함되어 있는 경우 '예'로 응답하세요.
            # 3. 문서에 질문에 관련되거나 유용한 정보가 포함되지 않은 경우 '아니오'로 응답합니다.
            # 4. 문서가 질문에 답할 수 있는 잠재적 가치를 제공하는지 여부에 초점을 맞추세요.
            # 5. 사소한 세부 사항이나 부정확성이 질문에 대한 문서의 전반적인 관련성에 큰 영향을 미치지 않는 한 무시하세요.

            # ### 질문:
            # {질문}

            # ### 검색된 문서:
            # {context1}

            # ### 작업:
            # 이진 점수를 제공합니다:
            # - 문서에 질문에 관련되거나 유용한 정보가 포함된 경우 '예'라고 합니다.
            # - 문서에 관련되거나 유용한 정보가 포함되지 않은 경우 '아니오'입니다.
            template = """
            You are a scorer tasked with evaluating whether a retrieved document is relevant to a given question.

            ### Instructions:
            1. Evaluate if the retrieved document contains information that could help answer or provide context to the question.
            2. If the document includes directly relevant or indirectly helpful information for the question, respond with 'yes'.
            3. If the document contains no relevant or useful information for the question, respond with 'no'.
            4. Focus on whether the document provides any potential value for answering the question, rather than strict factual matching.
            5. Ignore minor details or inaccuracies unless they significantly impact the document's overall relevance to the question.

            ### Question:
            {question}

            ### Retrieved Document:
            {context1}

            ### Task:
            Provide a binary score:
            - 'yes' if the document contains relevant or helpful information for the question.
            - 'no' if the document does not contain relevant or helpful information.
            """
            input_vars = ["question", "context1"]
            
        elif self.target == "score-question-retrieval":
            # 당신은 검색된 문서가 주어진 질문과 관련이 있는지 평가하는 채점자입니다.\n
            # 여기 질문이 있습니다:\n\n {질문} \n\n
            # 다음은 검색된 문서입니다:\n\n {context1} \n
            # 문서에 질문에 답하는 데 도움이 될 수 있는 정보가 포함되어 있으면 관련성이 있는 것으로 평가합니다.\n
            # 질문에 답하기 위한 의미론적 의미와 잠재적 유용성을 모두 고려하세요.\n

            # 검색된 문서가 질문과 관련이 있는지 여부를 나타내기 위해 이진 점수 '예' 또는 '아니오'를 제공합니다.
            template = """You are a grader assessing whether a retrieved document is relevant to the given question. \n
                Here is the question: \n\n {question} \n\n
                Here is the retrieved document: \n\n {context1} \n
                If the document contains information that could help answer the question, grade it as relevant. \n
                Consider both semantic meaning and potential usefulness for answering the question. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the question."""
            input_vars = ["question", "context1"]
        
        elif self.target == "summary-question-retrieval":
            # 당신은 검색된 문서가 주어진 질문과 관련이 있는지 평가하는 채점자입니다.\n
            # 여기 질문이 있습니다:\n\n {질문} \n\n
            # 다음은 검색된 문서입니다:\n\n {context1}, {context2} \\n
            # 문서에 질문에 답하는 데 도움이 될 수 있는 정보가 포함되어 있으면 관련성이 있는 것으로 평가합니다.\n
            # 질문에 답하기 위한 의미론적 의미와 잠재적 유용성을 모두 고려하세요.\n

            # 검색된 문서가 질문과 관련이 있는지 여부를 나타내기 위해 이진 점수 '예' 또는 '아니오'를 제공합니다.
            template = """You are a grader assessing whether a retrieved document is relevant to the given question. \n
                Here is the question: \n\n {question} \n\n
                Here is the retrieved document: \n\n {context1},{context2} \n
                If the document contains information that could help answer the question, grade it as relevant. \n
                Consider both semantic meaning and potential usefulness for answering the question. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the question."""
            input_vars = ["question", "context1",'context2']
                    
        elif self.target == "score-fact-check":
            # 당신은 평가 근거에 지원자의 자기소개서 내용과 회사 평가 기준 내용이 포함되어 있는지, 그리고 포함된 내용이 사실인지를 판단하기 위해 설계된 기계입니다.

            # 다음은 지원자의 자기소개서입니다:\n\n {original_document1} \\n\n
            # 회사 평가 기준은 다음과 같습니다:\n\n {original_document2} \\n\n
            # 평가 근거는 다음과 같습니다:\n\n {eval_document} \n
            # ## 지침:
            # 1. 제공된 평가 근거에 지원자의 자기소개서에서 파생된 내용이 포함되어 있는지 평가합니다.
            # 2. 제공된 평가 근거에 지원자의 자기소개서 내용{original_document1}이 없다면 회사 평가 기준{original_document2}이 포함되어 있는지 평가합니다.
            # 3. 평가 근거에 자기소개서에 사실적이고 진실한 내용이 포함된 경우 "예"로 응답하세요.
            # 4. 평가 근거에 회사 평가 기준에 사실적이고 진실한 내용이 포함된 경우 "예"로 응답하세요.
            # 5. 평가 근거에 자기소개서의 내용이 포함되어 있지만 허위 또는 오해의 소지가 있는 정보가 포함된 경우 "아니오"로 응답합니다.
            # 6. 평가 근거에 회사 평가 기준의 내용이 포함되어 있지만 허위 또는 오해의 소지가 있는 정보가 포함된 경우 "아니오"로 응답합니다.
            # 7. 평가 근거에 자기소개서의 내용과 회사 평가 기준 내용이 포함되지 않은 경우 "아니오"로 응답합니다.
            # 5. 평가 근거와 지원자의 자기소개서, 회사 평가 기준 내용 간의 사실적 일치에만 집중하세요
            template = """
                You are a machine designed to determine whether the evaluation basis includes content from the applicant's cover letter or the company's evaluation criteria, and whether the included content is factual.

                Here is the applicant's cover letter: \n\n {original_document1} \n\n
                Here is the company evaluation criteria: \n\n {original_document2} \n\n
                Here is the evaluation basis: \n\n {eval_document} \n

                ## Instructions:
                1. Evaluate whether the provided evaluation basis({eval_document})  contains content derived from the applicant's cover letter.
                2. If the evaluation basis({eval_document})  does not include content from the cover letter ({original_document1}), check if it includes content from the company's evaluation criteria ({original_document2}).
                3. If the evaluation basis includes factual and truthful content from the {original_document1}, respond with "yes."
                4. If the evaluation basis({eval_document})  includes factual and truthful content from the {original_document2}, respond with "yes."
                5. If the evaluation basis({eval_document})  includes content from the {original_document1} but contains false or misleading information, respond with "no."
                6. If the evaluation basis({eval_document})  includes content from the {original_document2} but contains false or misleading information, respond with "no."
                7. If the evaluation basis({eval_document})  does not include any content from the {original_document1} or the {original_document2}, respond with "no."
                8. Focus solely on factual alignment between the evaluation basis, the {original_document1}, and the company's evaluation criteria.
            """
            input_vars = ["original_document1","original_document2", "eval_document"]

        elif self.target == "question-fact-check":
            # 당신은 검색된 문서가 주어진 질문과 관련이 있는지 평가하는 채점자입니다.\n
            # 다음은 원본 문서입니다:\n\n {오리지널_document_1} \n\n {오리지널_document_2} \n
            # 생성된 질문은 다음과 같습니다:\n\n {질문} \n
            # 문서에 질문에 답하는 데 도움이 될 수 있는 정보가 포함되어 있으면 관련성이 있는 것으로 평가합니다.\n
            # 질문에 답하기 위한 의미론적 의미와 잠재적 유용성을 모두 고려하세요.\n

            # 검색된 문서가 질문과 관련이 있는지 여부를 나타내기 위해 이진 점수 '예' 또는 '아니오'를 부여합니다
            template = """
            You are a grader assessing whether a retrieved document is relevant to the given question. \n
            Here are the original documents: \n\n {original_document_1} \n\n {original_document_2} \n
            Here is the generated question: \n\n {question} \n
            If the document contains information that could help answer the question, grade it as relevant. \n
            Consider both semantic meaning and potential usefulness for answering the question. \n
            
            Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the question
            """
            input_vars = ["original_document_1", "original_document_2", "question"]
            
        
        elif self.target == "summary-fact-check":
            # 요약된 문서가 원본 문서와 비교했을 때 실제로 맞는지 확인하는 임무를 맡았습니다.

            # 다음은 원본 문서입니다: {original_document}
            # 요약된 문서는 다음과 같습니다: {summarized_document}

            # 당신의 임무는 두 문서를 비교하고 요약이 원본 문서의 사실을 정확하게 반영하는지 확인하는 것입니다.
            # - 요약이 사실적으로 맞고 원본 문서의 내용을 정확하게 나타낸다면, '예'라고 답하세요.
            # - 요약에 원본 문서와 모순되는 부정확하거나 오해의 소지가 있는 정보가 포함된 경우 '아니오'로 반환합니다.
            template = """
            You are tasked with checking whether the summarized document is factually correct compared to the original document.

            Here is the original document:  {original_document}
            Here is the summarized document: {summarized_document}

            Your task is to compare the two documents and determine if the summary accurately reflects the facts in the original document.
            - If the summary is factually correct and accurately represents the content of the original document, return 'yes'.
            - If the summary contains any incorrect or misleading information that contradicts the original document, return 'no'.
            """
            input_vars = ["original_document", "summarized_document"]

        else:
            raise ValueError(f"Invalid target: {self.target}")

        # 프롬프트 생성
        prompt = PromptTemplate(
            template=template,
            input_variables=input_vars,
        )

        # 체인
        chain = prompt | llm
        return chain