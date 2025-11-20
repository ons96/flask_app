// Simple JavaScript that works on older browsers (no ES6)
(function() {
    var chatMessages = document.getElementById('chat-messages');
    var messageInput = document.getElementById('message-input');
    var chatForm = document.getElementById('chat-form');
    var sendBtn = document.getElementById('send-btn');
    var clearBtn = document.getElementById('clear-btn');
    var providerSelect = document.getElementById('provider');
    var modelSelect = document.getElementById('model');
    
    // Load chat history from localStorage
    function loadHistory() {
        try {
            var history = localStorage.getItem('chatHistory');
            if (history) {
                chatMessages.innerHTML = history;
                scrollToBottom();
            }
        } catch (e) {
            console.log('LocalStorage not available');
        }
    }
    
    // Save chat history to localStorage
    function saveHistory() {
        try {
            localStorage.setItem('chatHistory', chatMessages.innerHTML);
        } catch (e) {
            console.log('LocalStorage not available');
        }
    }
    
    // Add message to chat
    function addMessage(text, type) {
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + type;
        
        var label = document.createElement('div');
        label.className = 'message-label';
        label.textContent = type === 'user' ? 'You' : (type === 'bot' ? 'AI' : 'Error');
        
        var content = document.createElement('div');
        content.textContent = text;
        
        messageDiv.appendChild(label);
        messageDiv.appendChild(content);
        chatMessages.appendChild(messageDiv);
        
        scrollToBottom();
        saveHistory();
    }
    
    // Scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Send message
    function sendMessage() {
        var message = messageInput.value.trim();
        if (!message) return;
        
        addMessage(message, 'user');
        messageInput.value = '';
        messageInput.disabled = true;
        sendBtn.disabled = true;
        sendBtn.textContent = 'Sending...';
        
        var provider = providerSelect.value;
        var model = modelSelect.value;
        
        // Make API request
        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/chat', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        
        xhr.onload = function() {
            messageInput.disabled = false;
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            
            if (xhr.status === 200) {
                try {
                    var data = JSON.parse(xhr.responseText);
                    if (data.response) {
                        addMessage(data.response, 'bot');
                    } else if (data.error) {
                        addMessage('Error: ' + data.error, 'error');
                    }
                } catch (e) {
                    addMessage('Error parsing response', 'error');
                }
            } else {
                addMessage('HTTP Error: ' + xhr.status, 'error');
            }
            
            messageInput.focus();
        };
        
        xhr.onerror = function() {
            messageInput.disabled = false;
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            addMessage('Network error. Check your connection.', 'error');
            messageInput.focus();
        };
        
        xhr.send(JSON.stringify({
            message: message,
            provider: provider,
            model: model
        }));
    }
    
    // Clear chat
    function clearChat() {
        if (confirm('Clear all messages?')) {
            chatMessages.innerHTML = '';
            try {
                localStorage.removeItem('chatHistory');
            } catch (e) {}
        }
    }
    
    // Event listeners
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        sendMessage();
    });
    
    clearBtn.addEventListener('click', clearChat);
    
    // Load history on page load
    loadHistory();
    messageInput.focus();
})();
