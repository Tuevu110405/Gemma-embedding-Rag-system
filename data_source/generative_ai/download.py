import os
import wget

# Danh sách các bài báo khoa học cần tải về
file_links = [
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf"
    },
    {
        "title": "BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "url": "https://arxiv.org/pdf/1810.04805.pdf"
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "url": "https://arxiv.org/pdf/2201.11903.pdf"
    },
    {
        "title": "Denoising Diffusion Probabilistic Models",
        "url": "https://arxiv.org/pdf/2006.11239.pdf"
    },
    {
        "title": "Instruction Tuning for Large Language Models- A Survey",
        "url": "https://arxiv.org/pdf/2308.10792.pdf"
    },
    {
        "title": "Llama 2- Open Foundation and Fine-Tuned Chat Models",
        "url": "https://arxiv.org/pdf/2307.09288.pdf"
    }
]

SOURCE_STORAGE = r"data_source"

# Hàm kiểm tra xem file đã tồn tại trong thư mục hay chưa
def is_exist(file_link):
    """Kiểm tra sự tồn tại của file PDF dựa trên tiêu đề."""
    return os.path.exists(os.path.join(SOURCE_STORAGE,f"{file_link['title']}.pdf"))

# Vòng lặp để tải các file chưa có
for file_link in file_links:
    if not is_exist(file_link):
        print(f"Đang tải: {file_link['title']}...")
        wget.download(file_link["url"], out=(os.path.join(SOURCE_STORAGE,f"{file_link['title']}.pdf")))
    else:
        print(f"Đã tồn tại: {file_link['title']}.pdf")