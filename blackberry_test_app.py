from flask import Flask, request, session, redirect, url_for
import os
import json
import uuid
from datetime import datetime
import html

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')

# Simple chat storage
CHATS_FILE = 'blackberry_chats.json'

def load_chats():
    """Load chats from file"""
    try:
        if os.path.exists(CHATS_FILE):
            with open(CHATS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chats: {e}")
    return {}

def save_chats(chats):
    """Save chats to file"""
    try:
        with open(CHATS_FILE, 'w') as f:
            json.dump(chats, f, indent=2)
    except Exception as e:
        print(f"Error saving chats: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    chats = load_chats()
    
    # Simple model list for testing
    available_models = [
        "gpt-3.5-turbo",
        "gpt-4",
        "claude-3-sonnet",
        "llama-3-70b"
    ]
    
    # Set default model
    default_model = available_models[0]
    
    # Initialize or load current chat
    if 'current_chat' not in session or session['current_chat'] not in chats:
        current_time = datetime.now().isoformat()
        session['current_chat'] = str(uuid.uuid4())
        chats[session['current_chat']] = {
            "history": [],
            "model": default_model,
            "name": "New Chat",
            "created_at": current_time,
            "last_modified": current_time
        }
        save_chats(chats)
    
    current_chat = chats[session['current_chat']]
    current_model = current_chat.get("model", default_model)
    
    # Handle POST requests
    if request.method == 'POST':
        # Handle New Chat
        if 'new_chat' in request.form:
            session['current_chat'] = str(uuid.uuid4())
            chats[session['current_chat']] = {
                "history": [],
                "model": default_model,
                "name": "New Chat",
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat()
            }
            save_chats(chats)
            return redirect(url_for('index'))
        
        # Handle message submission
        prompt_from_input = request.form.get('prompt', '').strip()
        selected_model = request.form.get('model', default_model)
        
        if prompt_from_input:
            try:
                # Add user message to history
                current_time = datetime.now().isoformat()
                current_chat["history"].append({
                    "role": "user",
                    "content": prompt_from_input,
                    "timestamp": current_time
                })
                
                # Update model if changed
                current_chat["model"] = selected_model
                
                # Simple demo response (replace with actual LLM API call)
                demo_response = f"This is a demo response from {selected_model}. You said: '{prompt_from_input}'. The Flask app is working correctly on your BlackBerry Classic!"
                
                # Add assistant response
                current_chat["history"].append({
                    "role": "assistant",
                    "content": demo_response,
                    "model": selected_model,
                    "provider": "Demo",
                    "timestamp": current_time
                })
                
                current_chat["last_modified"] = current_time
                
                # Auto-name chat if it's new
                if current_chat.get("name") == "New Chat" and len(current_chat["history"]) >= 2:
                    first_user_msg = next((msg["content"] for msg in current_chat["history"] if msg["role"] == "user"), None)
                    if first_user_msg:
                        clean_prompt = ''.join(c for c in ' '.join(first_user_msg.split()[:6]) if c.isalnum() or c.isspace()).strip()
                        timestamp_str = datetime.fromisoformat(current_time).strftime("%b %d, %I:%M%p")
                        chat_name = f"{clean_prompt[:30]}... ({timestamp_str})" if clean_prompt else f"Chat ({timestamp_str})"
                        current_chat["name"] = chat_name
                
                save_chats(chats)
                
            except Exception as e:
                print(f"Error processing message: {e}")
                # Add error response
                current_time = datetime.now().isoformat()
                current_chat["history"].append({
                    "role": "assistant",
                    "content": f"Error processing your request: {str(e)}",
                    "model": selected_model,
                    "provider": "Error Handler",
                    "timestamp": current_time
                })
                save_chats(chats)
            
            return redirect(url_for('index'))
    
    # Prepare chat history HTML
    history_html = ""
    for msg in current_chat.get("history", []):
        role_display = html.escape(msg["role"].title())
        timestamp_str = msg.get("timestamp", "")
        try:
            timestamp_display = datetime.fromisoformat(timestamp_str).strftime("%I:%M:%S %p") if timestamp_str else "No Time"
        except ValueError:
            timestamp_display = "Invalid Time"
        
        content_display = html.escape(msg["content"])
        
        # Create message metadata
        metadata = []
        if msg["role"] == "assistant":
            if msg.get('model', 'N/A') != 'N/A':
                metadata.append(f"Model: {html.escape(msg.get('model', 'N/A'))}")
            if msg.get('provider', 'N/A') != 'N/A':
                metadata.append(f"Provider: {html.escape(msg.get('provider', 'N/A'))}")
        
        metadata_display = f"<small>{' | '.join(metadata)}</small>" if metadata else ""
        
        # Determine message class based on role
        message_class = "user-message" if msg['role'] == 'user' else "assistant-message"
        
        history_html += f'''<div class="message {message_class}">
                             <div style="margin-bottom: 4px;">
                                <b>{role_display}</b> - <small>{timestamp_display}</small>
                             </div>
                             {f'<div style="margin-bottom: 4px; font-size: 12px; color: #666;">{metadata_display}</div>' if metadata_display else ''}
                             <div style="white-space: pre-wrap; word-wrap: break-word;">{content_display}</div>
                           </div>'''
    
    # Prepare model dropdown options
    model_options_html = ''
    for model_name in available_models:
        is_selected = (model_name == current_model)
        selected_attr = "selected" if is_selected else ""
        model_options_html += f'<option value="{model_name}" {selected_attr}>{model_name}</option>'
    
    # Navigation and controls
    nav_links_html = '''
        <form method="post" style="display: inline;">
            <input type="submit" name="new_chat" value="New Chat" style="padding: 4px 8px; font-size: 12px;">
        </form>
    '''
    
    # BlackBerry Classic optimized HTML
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>LLM Chat Interface</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 0; 
            background-color: #fff; 
            font-size: 14px;
        }}
        
        #top-controls {{ 
            padding: 8px; 
            background-color: #f0f0f0; 
            border-bottom: 1px solid #ccc; 
        }}
        
        #message-container {{ 
            padding: 10px; 
            margin-bottom: 10px; 
            min-height: 200px;
        }}
        
        .message {{ 
            margin-bottom: 15px; 
            padding: 8px;
            border: 1px solid #ddd;
        }}
        
        .user-message {{ 
            background-color: #e6f3ff; 
            border-left: 3px solid #0066cc;
        }}
        
        .assistant-message {{ 
            background-color: #f0f8f0; 
            border-left: 3px solid #009900;
        }}
        
        #controls-container {{ 
            background-color: #f0f0f0; 
            border-top: 1px solid #ddd; 
            padding: 10px;
        }}
        
        #model-selector {{ 
            margin-bottom: 10px; 
        }}
        
        select {{ 
            width: 100%; 
            padding: 4px; 
            font-size: 14px; 
            border: 1px solid #ccc;
        }}
        
        textarea {{ 
            width: 100%; 
            box-sizing: border-box; 
            height: 60px; 
            font-size: 14px; 
            margin-bottom: 8px; 
            padding: 8px; 
            border: 1px solid #ccc;
            font-family: Arial, sans-serif;
        }}
        
        input[type="submit"] {{ 
            padding: 8px 16px; 
            font-size: 14px; 
            border: 1px solid #ccc; 
            cursor: pointer; 
            background-color: #0066cc; 
            color: white;
            margin-right: 5px;
        }}
        
        .button-row {{ 
            margin-top: 5px; 
        }}
    </style>
</head>
<body>
    <div id="top-controls">
        <div>{nav_links_html}</div>
    </div>
    
    <div id="message-container">{history_html}</div>
    
    <div id="controls-container">
        <div id="model-selector">
            <select name="model" form="chat-form">{model_options_html}</select>
        </div>
        
        <div id="input-area">
            <form id="chat-form" method="post">
                <textarea name="prompt" placeholder="Type your message..."></textarea>
                <div class="button-row">
                    <input type="submit" name="send" value="Send">
                </div>
            </form>
        </div>
    </div>
</body>
</html>'''

if __name__ == '__main__':
    print("Starting BlackBerry-optimized Flask app...")
    app.run(host='0.0.0.0', port=5001, debug=True)