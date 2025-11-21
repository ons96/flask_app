// ES5 compatible version for older browsers like BlackBerry
(function() {
    'use strict';
    
    var models = {
        groq: ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
        openrouter: ['meta-llama/llama-3.1-70b-instruct:free', 'meta-llama/llama-3.1-8b-instruct:free', 'google/gemma-2-9b-it:free'],
        puter: ['claude-3.5-sonnet', 'gpt-4o', 'llama-3.1-70b', 'llama-3.1-405b']
    };
    
    // Get elements
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
    
    // Initialize on page load
    window.addEventListener('load', function() {
        if (providerSelect) updateModelDropdown();
        setupEventListeners();
    });
    
    function setupEventListeners() {
        if (providerSelect) {
            providerSelect.addEventListener('change', updateModelDropdown);
        }
        
        if (fileBtn) fileBtn.addEventListener('click', showFileUpload);
        if (urlBtn) urlBtn.addEventListener('click', showUrlInput);
        if (jsonBtn) jsonBtn.addEventListener('click', showJsonInput);
        if (fileUpload) fileUpload.addEventListener('change', loadCharacterFromFile);
        if (loadBtn) loadBtn.addEventListener('click', loadCurrentInput);
        if (sendBtn) sendBtn.addEventListener('click', sendMessage);
        if (regenBtn) regenBtn.addEventListener('click', regenerateLastMessage);
        if (clearBtn) clearBtn.addEventListener('click', clearChat);
        
        if (userInput) {
            userInput.addEventListener('keypress', function(e) {
                if (e.keyCode === 13 && !e.shiftKey && !userInput.disabled) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }
    }
    
    function showFileUpload() {
        loadMode = 'file';
        hideAllInputs();
        if (fileUpload) fileUpload.click();
    }
    
    function showUrlInput() {
        loadMode = 'url';
        hideAllInputs();
        if (urlInput) urlInput.style.display = 'block';
        if (loadBtn) loadBtn.style.display = 'inline-block';
        if (urlInput) urlInput.focus();
    }
    
    function showJsonInput() {
        loadMode = 'json';
        hideAllInputs();
        if (jsonInput) jsonInput.style.display = 'block';
        if (loadBtn) loadBtn.style.display = 'inline-block';
        if (jsonInput) jsonInput.focus();
    }
    
    function hideAllInputs() {
        if (urlInput) urlInput.style.display = 'none';
        if (jsonInput) jsonInput.style.display = 'none';
        if (loadBtn) loadBtn.style.display = 'none';
    }
    
    function updateModelDropdown() {
        if (!modelSelect || !providerSelect) return;
        var provider = providerSelect.value || 'openrouter';
        var modelList = models[provider] || ['meta-llama/llama-3.1-70b-instruct:free'];
        modelSelect.innerHTML = '';
        
        for (var i = 0; i < modelList.length; i++) {
            var option = document.createElement('option');
            option.value = modelList[i];
            option.textContent = modelList[i];
            modelSelect.appendChild(option);
        }
    }
    
    function loadCharacterFromFile(e) {
        if (!e.target.files || e.target.files.length === 0) return;
        var file = e.target.files[0];
        var reader = new FileReader();
        reader.onload = function(evt) {
            if (file.name.indexOf('.json') > -1) {
                try {
                    var character = JSON.parse(evt.target.result);
                    setCharacter(character);
                } catch (err) {
                    alert('Invalid JSON file: ' + err.message);
                }
            } else {
                alert('PNG character card support coming soon. Use JSON files for now.');
            }
        };
        reader.readAsText(file);
    }
    
    function loadCurrentInput() {
        if (loadMode === 'url' && urlInput) {
            loadCharacterFromURL(urlInput.value.trim());
        } else if (loadMode === 'json' && jsonInput) {
            loadCharacterFromJSON(jsonInput.value.trim());
        }
    }
    
    function loadCharacterFromURL(url) {
        if (!url) {
            alert('Please enter a URL');
            return;
        }
        if (chatContainer) addMessage('Loading character from URL...', 'loading');
        
        var payload = { url: url };
        callAPI('/api/character/parse', payload, function(data, error) {
            if (chatContainer) chatContainer.innerHTML = '';
            if (error) {
                alert('Failed to load character: ' + error);
            } else if (data && data.character) {
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
        if (userInput) userInput.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        if (regenBtn) regenBtn.disabled = false;
        var charName = character.name || (character.data && character.data.name) || 'Character';
        if (userInput) userInput.placeholder = 'Chat with ' + charName + '...';
        conversationHistory = [];
        if (chatContainer) chatContainer.innerHTML = '';
        addMessage('Character loaded: ' + charName, 'assistant');
        hideAllInputs();
    }
    
    // Rest of the roleplay functions (sendMessage, regenerate, etc.) remain the same as before
    function sendMessage() {
        if (!currentCharacter) {
            alert('Please load a character first');
            return;
        }
        
        if (!userInput) return;
        var message = userInput.value.trim();
        if (!message) return;
        
        var provider = providerSelect ? providerSelect.value : 'openrouter';
        var model = modelSelect ? modelSelect.value : 'meta-llama/llama-3.1-70b-instruct:free';
        
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
            lastMsg.parentNode.removeChild(lastMsg);
        }
        
        var lastUserMsg = conversationHistory[conversationHistory.length - 1].content;
        var provider = providerSelect ? providerSelect.value : 'openrouter';
        var model = modelSelect ? modelSelect.value : 'meta-llama/llama-3.1-70b-instruct:free';
        
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
        if (typeof puter === 'undefined' || !puter || !puter.ai || !puter.ai.chat) {
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
        if (chatContainer) chatContainer.innerHTML = '';
        conversationHistory = [];
    }
    
    function addMessage(text, type) {
        if (!chatContainer) return null;
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
        if (msg && msg.parentNode) {
            msg.parentNode.removeChild(msg);
        }
    }
    
    function disableInput(disabled) {
        if (userInput) userInput.disabled = disabled;
        if (sendBtn) sendBtn.disabled = disabled;
        if (regenBtn) regenBtn.disabled = disabled;
    }
    
    function callAPI(endpoint, payload, callback) {
        var xhr = new XMLHttpRequest();
        if (!xhr) return;
        
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
        
        if (xhr.ontimeout) {
            xhr.ontimeout = function() {
                callback(null, 'Request timed out');
            };
        }
        
        try {
            xhr.send(JSON.stringify(payload));
        } catch (e) {
            callback(null, 'Failed to send request');
        }
    }
})();
