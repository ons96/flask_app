(function() {
    var chatContainer = document.getElementById('chat-container');
    var userInput = document.getElementById('user-input');
    var sendBtn = document.getElementById('send-btn');
    var clearBtn = document.getElementById('clear-btn');
    var providerSelect = document.getElementById('provider-select');
    var modelInput = document.getElementById('model-input');
    var customEndpoint = document.getElementById('custom-endpoint');
    var customKey = document.getElementById('custom-key');
    var customFields = document.getElementById('custom-fields');
    
    var chatHistory = [];
    
    loadChatHistory();
    
    providerSelect.addEventListener('change', function() {
        customFields.style.display = providerSelect.value === 'custom' ? 'block' : 'none';
        if (providerSelect.value === 'groq') {
            modelInput.value = 'llama-3.1-70b-versatile';
        } else if (providerSelect.value === 'openrouter') {
            modelInput.value = 'meta-llama/llama-3.1-70b-instruct:free';
        }
    });
    
    sendBtn.addEventListener('click', sendMessage);
    clearBtn.addEventListener('click', clearChat);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.keyCode === 13 && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    function sendMessage() {
        var message = userInput.value.trim();
        if (!message) return;
        
        var provider = providerSelect.value;
        var model = modelInput.value.trim();
        
        addMessage(message, 'user');
        chatHistory.push({ role: 'user', content: message });
        userInput.value = '';
        
        var loadingId = addMessage('Thinking...', 'loading');
        disableInput(true);
        
        var payload = {
            messages: chatHistory,
            provider: provider,
            model: model
        };
        
        if (provider === 'custom') {
            payload.custom_endpoint = customEndpoint.value.trim();
            payload.custom_key = customKey.value.trim();
        }
        
        callAPI('/api/chat', payload, function(data, error) {
            removeMessage(loadingId);
            
            if (error) {
                addMessage('Error: ' + error, 'error');
            } else {
                addMessage(data.response, 'assistant');
                chatHistory.push({ role: 'assistant', content: data.response });
                saveChatHistory();
            }
            
            disableInput(false);
        });
    }
    
    function clearChat() {
        if (!confirm('Clear all messages?')) return;
        chatContainer.innerHTML = '';
        chatHistory = [];
        saveChatHistory();
    }
    
    function addMessage(text, type) {
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + type;
        messageDiv.id = 'msg-' + Date.now();
        messageDiv.textContent = text;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageDiv.id;
    }
    
    function removeMessage(id) {
        var msg = document.getElementById(id);
        if (msg) msg.remove();
    }
    
    function disableInput(disabled) {
        userInput.disabled = disabled;
        sendBtn.disabled = disabled;
        sendBtn.textContent = disabled ? 'Sending...' : 'Send';
    }
    
    function saveChatHistory() {
        try {
            localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        } catch (e) {}
    }
    
    function loadChatHistory() {
        try {
            var stored = localStorage.getItem('chatHistory');
            if (stored) {
                chatHistory = JSON.parse(stored);
                for (var i = 0; i < chatHistory.length; i++) {
                    var msg = chatHistory[i];
                    addMessage(msg.content, msg.role === 'user' ? 'user' : 'assistant');
                }
            }
        } catch (e) {}
    }
    
    function callAPI(endpoint, payload, callback) {
        var xhr = new XMLHttpRequest();
        xhr.open('POST', endpoint, true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.timeout = 90000;
        
        xhr.onload = function() {
            if (xhr.status === 200) {
                try {
                    var data = JSON.parse(xhr.responseText);
                    callback(data, null);
                } catch (e) {
                    callback(null, 'Failed to parse response');
                }
            } else {
                try {
                    var error = JSON.parse(xhr.responseText);
                    callback(null, error.error || 'HTTP ' + xhr.status);
                } catch (e) {
                    callback(null, 'HTTP ' + xhr.status);
                }
            }
        };
        
        xhr.onerror = function() {
            callback(null, 'Network error');
        };
        
        xhr.ontimeout = function() {
            callback(null, 'Request timed out');
        };
        
        xhr.send(JSON.stringify(payload));
    }
})();
