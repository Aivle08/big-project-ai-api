from pydantic import Field, BaseModel
from typing import List

class PDFInfo(BaseModel):
    """
    PDFDTO (PDF Data Transfer Object)

    PDF를 S3로 저장하기 위한 데이터 객체.

    Attributes:
        pdf_name (str): PDF파일명.
        applicant_id (int): 지원자의 고유 ID.
    """
    pdf_name: str
    applicant_id: int

class ResumeInsertDTO(BaseModel):
    """
    ResumeInsertDTO (ResumeInsert Data Transfer Object)

    지원자 자소서를 zilliz에 삽입하기 위한 데이터 객체.

    Attributes:
        pdf_info_list (List): 지원자 자소서. 
    """
    pdf_info_list: List[PDFInfo] = Field([PDFInfo(pdf_name='1_5.pdf', applicant_id=1000)], description='PDF 정보 리스트')
class EvalInsertDTO(BaseModel):
    """
    EvalInsertDTO (EvalInsert Data Transfer Object)

    평가기준 DB를 zilliz 저장하기 위한 데이터 객체.

    Attributes:
        recruitment_id (int): 공고 고유 ID
        detail_list (list): 평가 항목 내용 리스트.
    """
    recruitment_id: int = Field(1, description='공고 id')
    detail_list: list = Field(['a', 'b', 'c'], description='평가 항목 상세')

class ResumeDeleteDTO(BaseModel):
    """
    ResumeDeleteDTO (ResumeDelete Data Transfer Object)

    자소서를 삭제하기 위한 데이터 객체.

    Attributes:
        applicant_id_list (list): 지원자 고유 ID 리스트.
    """
    applicant_id_list: list = Field([1000], description='지원자 id')

class EvalDeleteDTO(BaseModel):
    """
    EvalDeleteDTO (EvalDelete Data Transfer Object)

    회사 평가 데이터를 삭제하기 위한 데이터 객체.

    Attributes:
        recruitment_id (int): 회사 공고 id.

    """
    recruitment_id: int = Field(1000, description='공고 id')