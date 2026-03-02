import fitz 
import io

def parse_pdf(file_contents: bytes) -> str:
    text_content = ""
    try:
        pdf_document = fitz.open(stream=file_contents, filetype="pdf")

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_content += page.get_text()

        pdf_document.close()
        return text_content

    except Exception as e:
        print(f"Error saat parsing PDF: {e}")
        return f"Error: Gagal memproses file PDF. Detail: {str(e)}"