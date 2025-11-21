(function() {
    var models = {
        groq: ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
        openrouter: ['meta-llama/llama-3.1-70b-instruct:free', 'meta-llama/llama-3.1-8b-instruct:free', 'google/gemma-2-9b-it:free'],
        puter: ['claude-3.5-sonnet', 'gpt-4o', 'llama-3.1-70b', 'llama-3.1-405b']
    };
    
    var chatContainer = document.getElementById('chat-container');
    var userInput = document.getElementById('user-input');
    var sendBtn = document.getElementById('send-btn');
    var regenBtn = document.getElementById('regen-btn');
    var clearBtn = document.getElementById('clear-btn');
    var providerSelect = document.getElementById('provider-select');
    var modelSelect = document.getElementById('model-select');
    
    var fileBtn = document.getElementById('file-btn');
    var urlBtn = document.getElementById('url-btn');
    var jsonBtn = document.getElementById('json-btn');
    var fileUpload = document.getElementById('file-upload');
    var urlInput = document.getElementById('url-input');
    var jsonInput = document.getElementById('json-input');
    var loadBtn = document.getElementById('load-btn');
    
    var currentCharacter = null;
    var conversationHistory = [];
    var loadMode = null;
    
    updateModelDropdown();
    
    providerSelect.addEventListener('change', updateModelDropdown);
    
    fileBtn.addEventListener('click', function() {
        loadMode = 'file';
        hideAllInputs();
        fileUpload.click();
    });
    
    urlBtn.addEventListener('click', function() {
        loadMode = 'url';
        hideAllInputs();
        urlInput.style.display = 'block';
        loadBtn.style.display = 'inline-block';
    });
    
    jsonBtn.addEventListener('click', function() {
        loadMode = 'json';
        hideAllInputs();
        jsonInput.style.display = 'block';
        loadBtn.style.display = 'inline-block';
    });
    
    fileUpload.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            loadCharacterFromFile(e.target.files[0]);
        }
    });
    
    loadBtn.addEventListener('click', function() {
        if (loadMode === 'url') {
            loadCharacterFromURL(urlInput.value.trim());
        } else if (loadMode === 'json') {
            loadCharacterFromJSON(jsonInput.value.trim());
        }
    });
    
    sendBtn.addEventListener('click', sendMessage);
    regenBtn.addEventListener('click', regenerateLastMessage);
    clearBtn.addEventListener('click', clearChat);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.keyCode === 13 && !e.shiftKey && !userInput.disabled) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    function hideAllInputs() {
        urlInput.style.display = 'none';
        jsonInput.style.display = 'none';
        loadBtn.style.display = 'none';
    }
    
    function updateModelDropdown() {
        var provider = providerSelect.value;
        var modelList = models[provider] || [];
        modelSelect.innerHTML = '';
        for (var i = 0; i < modelList.length; i++) {
            var opt = document.createElement('option');
            opt.value = modelList[i];
            opt.textContent = modelList[i];
            modelSelect.appendChild(opt);
        }
    }
    
    function loadCharacterFromFile(file) {
        var reader = new FileReader();
        reader.onload = function(e) {
            if (file.name.endsWith('.json')) {
                try {
                    var character = JSON.parse(e.target.result);
                    setCharacter(character);
                } catch (err) {
                    alert('Invalid JSON file');
                }
            } else if (file.name.endsWith('.png')) {
                alert('PNG character card parsing not yet implemented. Use JSON or URL instead.');
            }
        };
        reader.readAsText(file);
    }
    
    function loadCharacterFromURL(url) {
        if (!url) {
            alert('Please enter a URL');
            return;
        }
        addMessage('Loading character from URL...', 'loading');
        callAPI('/api/character/parse', { url: url }, function(data, error) {
            chatContainer.innerHTML = '';
            if (error) {
                alert('Failed to load character: ' + error);
            } else {
                setCharacter(data.character);
            }
        });
    }
    
    function loadCharacterFromJSON(jsonText) {
        if (!jsonText) {
            alert('Please paste character JSON');
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
        regenBtn.disabled = false;
        var charName = character.name || character.data && character.data.name || 'Character';
        userInput.placeholder = 'Chat with ' + charName + '...';
        conversationHistory = [];
        chatContainer.innerHTML = '';
        addMessage('Character loaded: ' + charName, 'assistant');
        hideAllInputs();
    }
    
    function sendMessage() {
        if (!currentCharacter) {
            alert('Please load a character first');
            return;
        }
        
        var message = userInput.value.trim();
        if (!message) return;
        
        var provider = providerSelect.value;
        var model = modelSelect.value;
        
        addMessage(message, 'user');
        conversationHistory.push({ role: 'user', content: message });
        userInput.value = '';
        
        var loadingId = addMessage('Typing...', 'loading');
        disableInput(true);
        
        if (provider === 'puter') {
            callPuterRoleplayAPI(message, function(response, error) {
                removeMessage(loadingId);
                if (error) {
                    addMessage('Error: ' + error, 'error');
                } else {
                    addMessage(response, 'assistant');
                    conversationHistory.push({ role: 'assistant', content: response });
                }
                disableInput(false);
            });
        } else {
            var payload = { message: message, character: currentCharacter, history: conversationHistory, provider: provider, model: model };
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
    }
    
    function regenerateLastMessage() {
        if (conversationHistory.length < 2 || conversationHistory[conversationHistory.length - 1].role !== 'assistant') {
            alert('No assistant message to regenerate');
            return;
        }
        
        conversationHistory.pop();
        var lastMsg = chatContainer.lastElementChild;
        if (lastMsg && lastMsg.classList.contains('assistant')) {
            lastMsg.remove();
        }
        
        var lastUserMsg = conversationHistory[conversationHistory.length - 1].content;
        var provider = providerSelect.value;
        var model = modelSelect.value;
        
        var loadingId = addMessage('Regenerating...', 'loading');
        disableInput(true);
        
        if (provider === 'puter') {
            callPuterRoleplayAPI(lastUserMsg, function(response, error) {
                removeMessage(loadingId);
                if (error) {
                    addMessage('Error: ' + error, 'error');
                } else {
                    addMessage(response, 'assistant');
                    conversationHistory.push({ role: 'assistant', content: response });
                }
                disableInput(false);
            });
        } else {
            var payload = { message: lastUserMsg, character: currentCharacter, history: conversationHistory, provider: provider, model: model };
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
    }
    
    function callPuterRoleplayAPI(message, callback) {
        if (typeof puter === 'undefined' || !puter.ai || !puter.ai.chat) {
            callback(null, 'Puter AI not available');
            return;
        }
        
        var charData = currentCharacter.data || currentCharacter;
        var systemPrompt = 'You are roleplaying as ' + (charData.name || 'a character') + '. ' + (charData.description || '') + ' ' + (charData.personality || '');
        var fullPrompt = systemPrompt + '

User: ' + message + '
Assistant:';
        
        puter.ai.chat(fullPrompt, { model: modelSelect.value })
            .then(function(response) {
                callback(response, null);
            })
            .catch(function(err) {
                callback(null, err.message || 'Puter API error');
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
        regenBtn.disabled = disabled;
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
