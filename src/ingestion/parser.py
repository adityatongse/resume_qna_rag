"""CV file parser for PDF and DOCX formats"""
import logging
from pathlib import Path
from typing import Dict, Any
import PyPDF2
from docx import Document


logger = logging.getLogger(__name__)


class CVParser:
    """Parse CV files (PDF and DOCX) and extract text content"""
    
    def __init__(self):
        """Initialize the CV parser"""
        self.supported_formats = [".pdf", ".docx"]
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse CV file and extract text content
        
        Args:
            file_path: Path to the CV file
            
        Returns:
            Dictionary containing extracted text and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"CV file not found: {file_path}")
            raise FileNotFoundError(f"CV file not found: {file_path}")
        
        file_extension = path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_extension}")
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        logger.info(f"Parsing CV file: {file_path}")
        
        try:
            if file_extension == ".pdf":
                return self._parse_pdf(path)
            elif file_extension == ".docx":
                return self._parse_docx(path)
        except Exception as e:
            logger.error(f"Error parsing CV file: {str(e)}")
            raise
    
    def _parse_pdf(self, path: Path) -> Dict[str, Any]:
        """
        Parse PDF file and extract text
        
        Args:
            path: Path object to PDF file
            
        Returns:
            Dictionary with text and metadata
        """
        try:
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                text_content = []
                page_texts = []
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        text_content.append(page_text)
                        page_texts.append({
                            "page": page_num + 1,
                            "text": page_text
                        })
                
                full_text = "\n\n".join(text_content)
                
                logger.info(f"Successfully parsed PDF: {num_pages} pages, {len(full_text)} characters")
                
                return {
                    "text": full_text,
                    "metadata": {
                        "file_name": path.name,
                        "file_type": "pdf",
                        "num_pages": num_pages,
                        "char_count": len(full_text)
                    },
                    "page_texts": page_texts
                }
        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
            raise ValueError(f"Failed to parse PDF file: {str(e)}")
    
    def _parse_docx(self, path: Path) -> Dict[str, Any]:
        """
        Parse DOCX file and extract text
        
        Args:
            path: Path object to DOCX file
            
        Returns:
            Dictionary with text and metadata
        """
        try:
            doc = Document(path)
            
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            
            full_text = "\n\n".join(paragraphs)
            
            logger.info(f"Successfully parsed DOCX: {len(paragraphs)} paragraphs, {len(full_text)} characters")
            
            return {
                "text": full_text,
                "metadata": {
                    "file_name": path.name,
                    "file_type": "docx",
                    "num_paragraphs": len(paragraphs),
                    "char_count": len(full_text)
                },
                "paragraphs": paragraphs
            }
        except Exception as e:
            logger.error(f"Error parsing DOCX: {str(e)}")
            raise ValueError(f"Failed to parse DOCX file: {str(e)}")
    
    def validate_content(self, parsed_data: Dict[str, Any]) -> bool:
        """
        Validate that parsed content is not empty
        
        Args:
            parsed_data: Parsed CV data
            
        Returns:
            True if content is valid, False otherwise
        """
        text = parsed_data.get("text", "")
        
        if not text or len(text.strip()) < 100:
            logger.warning("Parsed CV content is too short or empty")
            return False
        
        return True


