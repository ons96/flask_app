(function() {
    'use strict';
    
    // DOM Elements
    var chatContainer = document.getElementById('chat-container');
    var userInput = document.getElementById('user-input');
    var sendBtn = document.getElementById('send-btn');
    var clearBtn = document.getElementById('clear-btn');
    var characterFile = document.getElementById('character-file');
    var characterUrl = document.getElementById('character-url');
    var characterJson = document.getElementById('character-json');
    var loadUrlBtn = document.getElementById('load-url-btn');
    var loadJsonBtn = document.getElementById('load-json-btn');
    var characterInfo = document.getElementById('character-info');
    var providerSelect = document.getElementById('provider-select');
    var modelInput = document.getElementById('model-input');
    var jailbreakSelect = document.getElementById('jailbreak-select');
    var customEndpoint = document.getElementById('custom-endpoint');
    var customKey = document.getElementById('custom-key');
    var customApiFields = document.getElementById('custom-api-fields');
    var toggleCharacter = document.getElementById('toggle-character');
    var characterPanel = document.getElementById('character-panel');
    var characterSetup = document.getElementById('character-setup');
    
    // State
    var currentCharacter = null;
    var conversationHistory = [];
    var isPuterReady = false;
    
    // Initialize
    setupEventListeners();
    checkPuterStatus();
    
    // ========================================================================
    // EVENT LISTENERS
    // ========================================================================
    
    function setupEventListeners() {
        sendBtn.addEventListener('click', sendMessage);
        clearBtn.addEventListener('click', clearChat);
        characterFile.addEventListener('change', handleFileUpload);
        loadUrlBtn.addEventListener('click', loadCharacterFromUrl);
        loadJsonBtn.addEventListener('click', loadCharacterFromJson);
        toggleCharacter.addEventListener('click', toggleCharacterPanel);
        
        userInput.addEventListener('keypress', function(e) {
            if (e.keyCode === 13 && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        providerSelect.addEventListener('change', function() {
            customApiFields.style.display = providerSelect.value === 'custom' ? 'block' : 'none';
        });
    }
    
    function toggleCharacterPanel() {
        if (characterSetup.className.indexOf('collapsed') > -1) {
            characterSetup.className = 'character-setup';
            toggleCharacter.textContent = '▲ Hide Character Setup';
        } else {
            characterSetup.className = 'character-setup collapsed';
            toggleCharacter.textContent = '📝 Character Setup';
        }
    }
    
    // ========================================================================
    // CHARACTER LOADING
    // ========================================================================
    
    function handleFileUpload(e) {
        var file = e.target.files[0];
        if (!file) return;
        
        var formData = new FormData();
        formData.append('file', file);
        
        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/character/parse', true);
        
        xhr.onload = function() {
            if (xhr.status === 200) {
                var data = JSON.parse(xhr.responseText);
                setCharacter(data.character);
            } else {
                alert('Failed to load character: ' + xhr.responseText);
            }
        };
        
        xhr.send(formData);
    }
    
    function loadCharacterFromUrl() {
        var url = characterUrl.value.trim();
        if (!url) {
            alert('Please enter a character URL');
            return;
        }
        
        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/character/parse', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        
        xhr.onload = function() {
            if (xhr.status === 200) {
                var data = JSON.parse(xhr.responseText);
                setCharacter(data.character);
            } else {
                alert('Failed to load character from URL');
            }
        };
        
        xhr.send(JSON.stringify({ url: url }));
    }
    
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
        
        // Extract character data
        var charData = character.data || character;
        var name = charData.name || 'Unknown';
        var description = charData.description || 'No description';
        
        // Show character info
        characterInfo.innerHTML = '<h3>✓ ' + name + '</h3><p>' + description.substring(0, 100) + '...</p>';
        characterInfo.style.display = 'block';
        
        // Enable chat
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.placeholder = 'Chat with ' + name + '...';
        
        // Clear conversation
        conversationHistory = [];
        chatContainer.innerHTML = '';
        
        // Add first message if available
        if (charData.first_mes) {
            addMessage(charData.first_mes, 'assistant');
            conversationHistory.push({ role: 'assistant', content: charData.first_mes });
        }
        
        // Collapse character setup
        characterSetup.className = 'character-setup collapsed';
        toggleCharacter.textContent = '📝 Character Setup';
    }
    
    // ========================================================================
    // CHAT FUNCTIONS
    // ========================================================================
    
    function sendMessage() {
        if (!currentCharacter) {
            alert('Please load a character first');
            return;
        }
        
        var message = userInput.value.trim();
        if (!message) return;
        
        var provider = providerSelect.value;
        var model = modelInput.value.trim();
        var jailbreak = jailbreakSelect.value;
        
        // Add user message
        addMessage(message, 'user');
        conversationHistory.push({ role: 'user', content: message });
        userInput.value = '';
        
        // Show loading
        var loadingId = addMessage('Typing...', 'loading');
        disableInput(true);
        
        // Use Puter.js if selected
        if (provider === 'puter') {
            if (!isPuterReady) {
                removeMessage(loadingId);
                addMessage('Puter.js not ready', 'error');
                disableInput(false);
                return;
            }
            
            callPuterRoleplay(model, message, function(response, error) {
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
            // Call backend API
            var payload = {
                message: message,
                character: currentCharacter,
                history: conversationHistory,
                provider: provider,
                model: model,
                jailbreak: jailbreak
            };
            
            if (provider === 'custom') {
                payload.custom_endpoint = customEndpoint.value.trim();
                payload.custom_key = customKey.value.trim();
            }
            
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
    
    function clearChat() {
        if (!confirm('Clear conversation?')) return;
        chatContainer.innerHTML = '';
        conversationHistory = [];
        
        // Add first message again if available
        if (currentCharacter) {
            var charData = currentCharacter.data || currentCharacter;
            if (charData.first_mes) {
                addMessage(charData.first_mes, 'assistant');
                conversationHistory.push({ role: 'assistant', content: charData.first_mes });
            }
        }
    }
    
    // ========================================================================
    // UI FUNCTIONS
    // ========================================================================
    
    function addMessage(text, type) {
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + type;
        messageDiv.id = 'msg-' + Date.now();
        messageDiv.textContent = text;
        
        chatContainer.appendChild(messageDiv);
        scrollToBottom();
        
        return messageDiv.id;
    }
    
    function removeMessage(id) {
        var msg = document.getElementById(id);
        if (msg) msg.remove();
    }
    
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    function disableInput(disabled) {
        userInput.disabled = disabled;
        sendBtn.disabled = disabled;
        sendBtn.textContent = disabled ? 'Sending...' : 'Send';
    }
    
    // ========================================================================
    // API CALLS
    // ========================================================================
    
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
    
    // ========================================================================
    // PUTER.JS INTEGRATION
    // ========================================================================
    
    function checkPuterStatus() {
        var checkInterval = setInterval(function() {
            if (typeof puter !== 'undefined' && puter.ai) {
                isPuterReady = true;
                console.log('Puter.js ready');
                clearInterval(checkInterval);
            }
        }, 100);
        
        setTimeout(function() {
            clearInterval(checkInterval);
        }, 10000);
    }
    
    function callPuterRoleplay(model, message, callback) {
        if (!puter || !puter.ai) {
            callback(null, 'Puter.js not available');
            return;
        }
        
        // Build context with character and history
        var context = buildPuterContext();
        var fullMessage = context + '

User: ' + message + '
Assistant:';
        
        try {
            puter.ai.chat(fullMessage, { model: model }).then(function(response) {
                callback(response, null);
            }).catch(function(error) {
                callback(null, error.message || 'Puter error');
            });
        } catch (e) {
            callback(null, e.message);
        }
    }
    
    function buildPuterContext() {
        var charData = currentCharacter.data || currentCharacter;
        var context = 'Character: ' + (charData.name || 'Unknown') + '
';
        context += 'Description: ' + (charData.description || '') + '
';
        context += 'Personality: ' + (charData.personality || '') + '

';
        context += 'Conversation:
';
        
        for (var i = 0; i < conversationHistory.length; i++) {
            var msg = conversationHistory[i];
            context += (msg.role === 'user' ? 'User' : 'Assistant') + ': ' + msg.content + '
';
        }
        
        return context;
    }
})();
