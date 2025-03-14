from flask import Flask, request, jsonify
from transformers import AutoTokenizer, BartTokenizer, BartForConditionalGeneration, AutoModelForQuestionAnswering, pipeline
import pdfplumber
import os
import torch

app = Flask(__name__)

# Load models and tokenizers
device = 0 if torch.cuda.is_available() else -1
summary_model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(summary_model_name)
summary_model = BartForConditionalGeneration.from_pretrained(summary_model_name).to("cuda" if device == 0 else "cpu")

chat_model_name = "deepset/roberta-base-squad2"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
chat_model = AutoModelForQuestionAnswering.from_pretrained(chat_model_name).to("cuda" if device == 0 else "cpu")

summary_pipeline = pipeline("summarization", model=summary_model, tokenizer=tokenizer, device=device)
chat_pipeline = pipeline("question-answering", model=chat_model, tokenizer=chat_tokenizer, device=device)

summarized_content = ""

def extract_text_from_pdf(file):
    """Extract text from the uploaded PDF."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error while extracting text: {e}")
        raise
    return text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    user_message = request.json.get("message", "")
    file_content = request.json.get("file_content", "")  # Get file content from request

    # Combine file content with default tax context
    default_context = """
    Income tax filing in India is an annual process where individuals, Hindu Undivided Families (HUFs), and businesses report their income and pay taxes to the Income Tax Department, governed by the Income Tax Act, 1961. The tax year, referred to as the financial year (FY), runs from April 1 to March 31, and the corresponding assessment year (AY) is when the income earned is assessed for tax purposes. Individuals are classified into different categories—resident, non-resident, and resident but not ordinarily resident (RNOR)—based on their residency status, and each category is taxed accordingly. Taxable income is divided into five heads: Salary, Income from House Property, Profits and Gains of Business or Profession, Capital Gains, and Income from Other Sources. Taxpayers can claim deductions under sections like 80C (e.g., for investments in Employee Provident Fund, Public Provident Fund, and ELSS), 80D (for health insurance premiums), and others, which reduce taxable income. Additionally, exemptions such as House Rent Allowance (HRA) and Leave Travel Allowance (LTA) under section 10 can be availed based on eligibility. The government provides different tax regimes: the old regime with multiple exemptions and deductions, and the new regime, which offers lower tax rates but fewer exemptions and deductions. Individuals whose annual income exceeds the basic exemption limit (₹2.5 lakh for individuals below 60, ₹3 lakh for senior citizens, and ₹5 lakh for super senior citizens) must file an Income Tax Return (ITR). ITR forms vary based on the taxpayer's income sources and status, ranging from ITR-1 (for salaried individuals) to ITR-7 (for trusts and institutions). Filing can be done online through the Income Tax Department’s e-filing portal or offline for select cases. Salaried individuals receive Form 16 from their employers, which summarizes salary income and TDS (Tax Deducted at Source). Similarly, businesses and freelancers must maintain books of accounts and compute taxes, availing benefits like presumptive taxation under sections 44AD or 44ADA for small businesses and professionals, respectively. Taxpayers are required to reconcile TDS with Form 26AS and AIS (Annual Information Statement), ensuring all income sources are correctly reported. Penalties are levied for non-compliance, such as failing to file returns on time or furnishing incorrect details. Advance tax payments must be made quarterly by individuals with substantial non-salaried income or businesses to avoid interest under sections 234B and 234C. Senior citizens without business income are exempt from paying advance tax. Refunds for excess taxes paid can be claimed during filing, and they are processed with interest under section 244A. Late filing attracts penalties of up to ₹5,000 under section 234F, and the last date for filing without penalty is typically July 31 of the assessment year. Revision of returns is allowed within a specified period if errors are found post-filing. The government also emphasizes digitalization and simplification of tax processes through pre-filled ITR forms and faceless assessments, reducing manual intervention and chances of disputes. High-value transactions, such as property purchases, stock trading, and cash deposits above certain limits, are closely monitored and must be declared. Individuals can use tax filing software or engage chartered accountants for accurate filing. Understanding tax implications on income from capital gains, dividends, or international income is critical, especially for NRIs, who are taxed only on income earned or accrued in India. Failure to comply with tax regulations can lead to scrutiny, penalties, or even prosecution under severe cases of tax evasion. Overall, income tax filing is a crucial civic duty that ensures financial accountability, contributes to nation-building, and enables individuals to stay compliant while optimizing their tax outgo through legal provisions.
    """
    combined_context = f"{file_content}\n\n{default_context}" if file_content else default_context

    # Log context for debugging
    #print(f"Context passed to the model:\n{combined_context}")

    # Prepare input for the model
    input_data = {
        "question": user_message,
        "context": combined_context,
    }

    try:
        response = chat_pipeline(input_data)
        return jsonify({"response": response.get('answer', 'No response found.')})
    except Exception as e:
        print(f"Error during chat generation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    global summarized_content

    file = request.files.get("file")

    if file:
        file_content = extract_text_from_pdf(file)

        # Split the text into manageable chunks
        chunk_size = 800  # Adjust based on your model's capacity
        chunks = list(split_into_chunks(file_content, chunk_size))

        # Summarize each chunk and combine the results
        summaries = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length = 1024)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            summary_ids = summary_model.generate(
                inputs["input_ids"], 
                max_length=150, 
                min_length=30, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

        # Combine all chunk summaries into a single string
        summarized_content = " ".join(summaries)
        return jsonify({"summary": summarized_content})
        
    return jsonify({"error": "No file uploaded"})


def split_into_chunks(text, chunk_size=900):
    """Split text into chunks of a specified size."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(port=5000)
