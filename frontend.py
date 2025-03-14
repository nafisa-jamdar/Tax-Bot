import subprocess
import requests
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox

# Start the Flask backend server as a subprocess
backend_process = subprocess.Popen(['python', 'backendTEST.py'])

# Define the Flask server URL
backend_url_chat = "http://127.0.0.1:5000/chat"
backend_url_upload = "http://127.0.0.1:5000/upload"

summarized_content = ""

def send_message():
    """Send a user message to the chat endpoint."""
    global summarized_content  # Use the global variable to fetch uploaded file content

    user_message = entry.get()
    entry.delete(0, tk.END)

    try:
        # Include summarized content in the payload if available
        data = {"message": user_message}
        if summarized_content:
            data["file_content"] = summarized_content

        # Send request to the chat endpoint
        response = requests.post(backend_url_chat, json=data)
        if response.status_code == 200:
            bot_response = response.json().get("response", "No response")
            chatbox.insert(tk.END, f"You: {user_message}\nBot: {bot_response}\n")
        else:
            chatbox.insert(tk.END, f"Error: {response.text}\n")
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Connection Error", f"Error connecting to the server:\n{e}")


def upload_file():
    """Upload a PDF file and process its content."""
    global summarized_content

    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if not file_path:
        return

    try:
        with open(file_path, "rb") as file:
            response = requests.post(backend_url_upload, files={"file": file})
            if response.status_code == 200:
                summarized_content = response.json().get("summary", "")
                chatbox.insert(tk.END, f"Uploaded file analyzed:\n{summarized_content}\n")
            else:
                chatbox.insert(tk.END, f"Error: {response.text}\n")
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Connection Error", f"Error connecting to the server:\n{e}")

# Create GUI components
root = tk.Tk()
root.title("TaxBot Chat")
root.geometry("500x500")

chatbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, state=tk.NORMAL)
chatbox.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)
entry.bind("<Return>", lambda event: send_message())  # Press Enter to send message

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=5)

upload_button = tk.Button(root, text="Upload PDF", command=upload_file)
upload_button.pack(pady=5)

# Clean up the backend process when the GUI is closed
def on_closing():
    backend_process.terminate()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()

