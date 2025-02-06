from pydantic import Field, BaseModel
from typing import List

class PDFInfo(BaseModel):
    pdf_name: str
    applicant_id: int

class ResumeInsertDTO(BaseModel):
    pdf_info_list: List[PDFInfo] = Field([PDFInfo(pdf_name='1_5.pdf', applicant_id=1000)], description='PDF 정보 리스트')

class EvalInsertDTO(BaseModel):
    recruitment_id: int = Field(1, description='공고 id')
    detail_list: list = Field(['a', 'b', 'c'], description='평가 항목 상세')

class ResumeDeleteDTO(BaseModel):
    applicant_id_list: list = Field([1000], description='지원자 id')

class EvalDeleteDTO(BaseModel):
    recruitment_id: int = Field(1000, description='공고 id')