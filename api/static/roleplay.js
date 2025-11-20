(function() {
    var chatContainer = document.getElementById('chat-container');
    var userInput = document.getElementById('user-input');
    var sendBtn = document.getElementById('send-btn');
    var clearBtn = document.getElementById('clear-btn');
    var characterJson = document.getElementById('character-json');
    var loadJsonBtn = document.getElementById('load-json-btn');
    var providerSelect = document.getElementById('provider-select');
    var modelInput = document.getElementById('model-input');
    
    var currentCharacter = null;
    var conversationHistory = [];
    
    loadJsonBtn.addEventListener('click', loadCharacterFromJson);
    sendBtn.addEventListener('click', sendMessage);
    clearBtn.addEventListener('click', clearChat);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.keyCode === 13 && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    function loadCharacterFromJson() {
        var jsonText = characterJson.value.trim();
        if (!jsonText) {
            alert('Please enter character JSON');
            return;
        }
        
        try {
            var character = JSON.parse(jsonText);
            setCharacter(character);
        } catch (e) {
            alert('Invalid JSON: ' + e.message);
        }
    }
    
    function setCharacter(character) {
        currentCharacter = character;
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.placeholder = 'Chat with ' + (character.name || 'character') + '...';
        conversationHistory = [];
        chatContainer.innerHTML = '';
        addMessage('Character loaded: ' + (character.name || 'Unknown'), 'assistant');
    }
    
    function sendMessage() {
        if (!currentCharacter) {
            alert('Please load a character first');
            return;
        }
        
        var message = userInput.value.trim();
        if (!message) return;
        
        var provider = providerSelect.value;
        var model = modelInput.value.trim();
        
        addMessage(message, 'user');
        conversationHistory.push({ role: 'user', content: message });
        userInput.value = '';
        
        var loadingId = addMessage('Typing...', 'loading');
        disableInput(true);
        
        var payload = {
            message: message,
            character: currentCharacter,
            history: conversationHistory,
            provider: provider,
            model: model
        };
        
        callAPI('/api/roleplay/chat', payload, function(data, error) {
            removeMessage(loadingId);
            
            if (error) {
                addMessage('Error: ' + error, 'error');
            } else {
                addMessage(data.response, 'assistant');
                conversationHistory.push({ role: 'assistant', content: data.response });
            }
            
            disableInput(false);
        });
    }
    
    function clearChat() {
        if (!confirm('Clear conversation?')) return;
        chatContainer.innerHTML = '';
        conversationHistory = [];
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
