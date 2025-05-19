import os
from flask import Flask, render_template, request, send_file
import pdfplumber
import docx
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from werkzeug.utils import secure_filename
import google.generativeai as genai
from fpdf import FPDF  # pip install fpdf

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDF0upkbUWE3ZuuNb21gcsVIYFpSMJKUIo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages])
        return text
    elif ext == 'docx':
        doc = docx.Document(file_path)
        text = ' '.join([para.text for para in doc.paragraphs])
        return text
    elif ext == 'txt':
        with open(file_path, 'r') as file:
            return file.read()
    return None

def summarize_text_with_bert(input_text, max_length=512):
    """Use BERT to summarize or extract key parts of the text."""
    inputs = bert_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # For simplicity, return the input text truncated to the BERT model's max token length
    return bert_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

def question_mcqs_generator(input_text, num_questions, difficulty, true_false_only):
    if true_false_only:
        prompt = f"""
        You are an AI assistant helping the user generate True/False questions based on the following text:
        '{input_text}'
        Please generate {num_questions} questions from the text with a difficulty level of '{difficulty}'.
        Each question should have:
        - A clear question
        - Two answer options (labeled A, B)
        - The correct answer clearly indicated
        Format:
        
        ## MCQ
        Question: [question]
        A) True
        B) False
        Correct Answer: [True/False]
        """
    else:
        prompt = f"""
        You are an AI assistant helping the user generate multiple-choice questions (MCQs) based on the following text:
        '{input_text}'
        Please generate {num_questions} MCQs from the text with a difficulty level of '{difficulty}'. Each question should have:
        - A clear question
        - Four answer options (labeled A, B, C, D)
        - The correct answer clearly indicated
        Format:
        ## MCQ
        Question: [question]
        A) [option A]
        B) [option B]
        C) [option C]
        D) [option D]
        Correct Answer: [correct option]
        """
    response = gemini_model.generate_content(prompt).text.strip()
    return response

def save_mcqs_to_file(mcqs, filename):
    results_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    with open(results_path, 'w') as f:
        f.write(mcqs)
    return results_path

def create_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("## MCQ"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)  # Add a line break

    pdf_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    pdf.output(pdf_path)
    return pdf_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_mcqs():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from the uploaded file
        text = extract_text_from_file(file_path)

        if text:
            # Use BERT to summarize or extract relevant content
            summarized_text = summarize_text_with_bert(text)

            # Generate MCQs with Gemini
            num_questions = int(request.form['num_questions'])
            difficulty = request.form['difficulty']
            true_false_only = 'question_type' in request.form and request.form['question_type'] == 'true_false'

            mcqs = question_mcqs_generator(summarized_text, num_questions, difficulty, true_false_only)

            # Save the generated MCQs to a file
            txt_filename = f"generated_mcqs_{filename.rsplit('.', 1)[0]}.txt"
            pdf_filename = f"generated_mcqs_{filename.rsplit('.', 1)[0]}.pdf"
            save_mcqs_to_file(mcqs, txt_filename)
            create_pdf(mcqs, pdf_filename)

            # Display and allow downloading
            return render_template('results.html', mcqs=mcqs, txt_filename=txt_filename, pdf_filename=pdf_filename)
    return "Invalid file format"

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RESULTS_FOLDER']):
        os.makedirs(app.config['RESULTS_FOLDER'])
    app.run(debug=True)
