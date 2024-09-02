import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import detected_emotion_global #import gv

class MyLLMModel:
    def __init__(self):
        MODEL_NAME = "../Meta-Llama-3-8B-Instruct"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quantization_config, device_map="auto"
        )
        
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=128,
            return_full_text=False,
        )

    def generate(self, user_input):
        messages = [
            {
                "role": "system",
                "content": f"You are an emotional support chatbot named Bu Shuang. You are able to help user solve mental health problem.Your speaking style is gentle and compassionate.The emotion of user is {detected_emotion_global.detected_emotion}",
            },
            {"role": "user", "content": user_input},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipe(prompt)
        response = outputs[0]['generated_text']
        return response
 
llm_model = MyLLMModel()

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Chatbot UI')
        self.setGeometry(100, 100, 400, 500)

        # Set main layout
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

        # History text area for conversation history
        self.historyEdit = QTextEdit()
        self.historyEdit.setFont(QFont('Arial', 10))
        self.historyEdit.setReadOnly(True)
        self.layout.addWidget(self.historyEdit)

        # Input label and text area
        self.label = QLabel('Question:')
        self.label.setFont(QFont('Arial', 12))
        self.layout.addWidget(self.label)
        
        self.textEdit = QTextEdit()
        self.textEdit.setFont(QFont('Arial', 10))
        self.textEdit.setFixedHeight(80)
        self.layout.addWidget(self.textEdit)
        
        # Horizontal layout for button
        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addStretch()

        # Send button
        self.button = QPushButton('Send')
        self.button.setFont(QFont('Arial', 12))
        self.button.setStyleSheet("QPushButton { background-color : lightblue; }")
        self.button.clicked.connect(self.on_click)
        self.buttonLayout.addWidget(self.button)

        # Add button layout to main layout
        self.layout.addLayout(self.buttonLayout)
        
        # Set main layout
        self.setLayout(self.layout)

    def on_click(self):
        user_input = self.textEdit.toPlainText()
        if user_input.strip():
            response = llm_model.generate(user_input)
            # Update history with user input and model response
            self.update_history(user_input, response)
            # Clear input text area after sending
            self.textEdit.clear()

    def update_history(self, user_input, response):
        # Append user input and response to the history
        history_text = f"<b>User:</b> {user_input}<br><b>Assistant:</b> {response}<br><br>"
        self.historyEdit.append(history_text)
def start_chat_app():
    app = QApplication(sys.argv)
    ex = ChatApp()
    ex.show()
    sys.exit(app.exec_())

