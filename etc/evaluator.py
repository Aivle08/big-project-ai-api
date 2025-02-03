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
        elif self.target == "summary-question-retrieval":
            llm = self.llm.with_structured_output(GroundnessQuestionRetrievalScore)
        elif self.target == "question-fact-check":
            llm = self.llm.with_structured_output(QuestionFactCheckScore)
        elif self.target == "summary-fact-check":
            llm = self.llm.with_structured_output(SummaryFactCheckScore)
        else:
            raise ValueError(f"Invalid target: {self.target}")

        # 프롬프트 선택
        if self.target == "retrieval-answer":
            template = """You are a grader assessing relevance of a retrieved document to a user question. \n 
                Here is the retrieved document: \n\n {context} \n\n
                Here is the answer: {answer} \n
                If the document contains keyword(s) or semantic meaning related to the user answer, grade it as relevant. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the answer."""
            input_vars = ["context", "answer"]

        elif self.target == "question-answer":
            template = """You are a grader assessing whether an answer appropriately addresses the given question. \n
                Here is the question: \n\n {question} \n\n
                Here is the answer: {answer} \n
                If the answer directly addresses the question and provides relevant information, grade it as relevant. \n
                Consider both semantic meaning and factual accuracy in your assessment. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the answer is relevant to the question."""
            input_vars = ["question", "answer"]

        elif self.target == "generate-question-retrieval":
            template = """
            You are a scorer who assesses whether a searched document is related to a given question.
            Here is the question: \n\n {question} \n\n
            Here is the retrieved document: \n\n {context1} \n
            You should also assess whether the retrieved document contains accurate information about the question,
            or if LLM is likely to hallucinate content that is not in the document.
            
            Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the question.
            """
            input_vars = ["question", "context1"]
            
        elif self.target == "summary-question-retrieval":
            template = """You are a grader assessing whether a retrieved document is relevant to the given question. \n
                Here is the question: \n\n {question} \n\n
                Here is the retrieved document: \n\n {context1} \n
                If the document contains information that could help answer the question, grade it as relevant. \n
                Consider both semantic meaning and potential usefulness for answering the question. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the question."""
            input_vars = ["question", "context1"]
        
        elif self.target == "question-fact-check":
            # 당신은 평가 기준에 지원자의 자기소개서 내용이 포함되어 있는지, 그리고 포함된 내용이 사실인지를 판단하기 위해 설계된 기계입니다.

            # 다음은 지원자의 자기소개서입니다:\n\n {오리지널_document} \n\n
            # 평가 기준은 다음과 같습니다:\n\n {eval_document} \n

            # ## 지침:
            # 1. 제공된 평가 기준에 지원자의 자기소개서에서 파생된 내용이 포함되어 있는지 평가합니다.
            # 2. 평가 근거에 자기소개서에 사실적이고 진실한 내용이 포함된 경우 "예"로 응답하세요.
            # 3. 평가 기준에 자기소개서의 내용이 포함되어 있지만 허위 또는 오해의 소지가 있는 정보가 포함된 경우 "아니오"로 응답합니다.
            # 4. 평가 기준에 자기소개서의 내용이 포함되지 않은 경우 "아니오"로 응답합니다.
            # 5. 평가 기준과 지원자의 자기소개서 내용 간의 사실적 일치에만 집중하세요.
            template = """
            You are a machine designed to determine whether the evaluation basis includes content from the applicant's cover letter and if the included content is factual.

            Here is the applicant's cover letter: \n\n {original_document} \n\n
            Here is the evaluation basis: \n\n {eval_document} \n
            
            ## Instructions:
            1. Evaluate whether the provided evaluation basis contains content derived from the applicant's cover letter.
            2. If the evaluation basis includes factual and truthful content from the cover letter, respond with "yes".
            3. If the evaluation basis includes content from the cover letter but contains false or misleading information, respond with "no".
            4. If the evaluation basis does not include any content from the cover letter, respond with "no".
            5. Focus solely on the factual alignment between the evaluation basis and the content in the applicant's cover letter.
            """
            input_vars = ["original_document", "eval_document"]

        elif self.target == "summary-fact-check":
            template = """
            You are a grader assessing whether a retrieved document is relevant to the given question. \n
            Here are the original documents: \n\n {original_document_1} \n\n {original_document_2} \n
            Here is the generated question: \n\n {question} \n
            If the document contains information that could help answer the question, grade it as relevant. \n
            Consider both semantic meaning and potential usefulness for answering the question. \n
            
            Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the question
            """
            input_vars = ["original_document_1", "original_document_2", "question"]


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