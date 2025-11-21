// ES5 compatible version for older browsers like BlackBerry
(function() {
    'use strict';
    
    var models = {
        groq: ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
        openrouter: ['meta-llama/llama-3.1-70b-instruct:free', 'meta-llama/llama-3.1-8b-instruct:free', 'google/gemma-2-9b-it:free', 'mistralai/mistral-7b-instruct:free', 'qwen/qwen-2-7b-instruct:free'],
        puter: ['claude-3.5-sonnet', 'gpt-4o', 'gpt-4o-mini', 'gemini-1.5-pro', 'mistral-large', 'llama-3.1-70b', 'llama-3.1-405b'],
        custom: ['custom-model']
    };
    
    // Get elements
    var chatContainer = document.getElementById('chat-container');
    var userInput = document.getElementById('user-input');
    var sendBtn = document.getElementById('send-btn');
    var regenBtn = document.getElementById('regen-btn');
    var clearBtn = document.getElementById('clear-btn');
    var providerSelect = document.getElementById('provider-select');
    var modelSelect = document.getElementById('model-select');
    var customEndpoint = document.getElementById('custom-endpoint');
    var customKey = document.getElementById('custom-key');
    var customFields = document.getElementById('custom-fields');
    
    var chatHistory = [];
    
    // Initialize on page load
    window.addEventListener('load', function() {
        loadChatHistory();
        updateModelDropdown();
        setupEventListeners();
    });
    
    function setupEventListeners() {
        if (!providerSelect) return;
        providerSelect.addEventListener('change', function() {
            customFields.style.display = providerSelect.value === 'custom' ? 'block' : 'none';
            updateModelDropdown();
        });
        
        if (sendBtn) sendBtn.addEventListener('click', sendMessage);
        if (regenBtn) regenBtn.addEventListener('click', regenerateLastMessage);
        if (clearBtn) clearBtn.addEventListener('click', clearChat);
        
        if (userInput) {
            userInput.addEventListener('keypress', function(e) {
                if (e.keyCode === 13 && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }
    }
    
    function updateModelDropdown() {
        if (!modelSelect || !providerSelect) return;
        
        var provider = providerSelect.value || 'groq';
        var modelList = models[provider] || ['default-model'];
        modelSelect.innerHTML = '';
        
        for (var i = 0; i < modelList.length; i++) {
            var option = document.createElement('option');
            option.value = modelList[i];
            option.textContent = modelList[i];
            modelSelect.appendChild(option);
        }
    }
    
    function sendMessage() {
        if (!userInput) return;
        var message = userInput.value.trim();
        if (!message) return;
        
        var provider = providerSelect ? providerSelect.value : 'groq';
        var model = modelSelect ? modelSelect.value : 'llama-3.1-70b-versatile';
        
        addMessage(message, 'user');
        chatHistory.push({ role: 'user', content: message });
        userInput.value = '';
        
        var loadingId = addMessage('Thinking...', 'loading');
        disableInput(true);
        
        if (provider === 'puter') {
            callPuterAPI(chatHistory, model, function(response, error) {
                removeMessage(loadingId);
                if (error) {
                    addMessage('Error: ' + error, 'error');
                } else {
                    addMessage(response, 'assistant');
                    chatHistory.push({ role: 'assistant', content: response });
                    saveChatHistory();
                }
                disableInput(false);
            });
        } else {
            var payload = { messages: chatHistory, provider: provider, model: model };
            if (provider === 'custom' && customEndpoint && customKey) {
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
    }
    
    function regenerateLastMessage() {
        if (chatHistory.length < 2) {
            alert('No message to regenerate');
            return;
        }
        
        if (chatHistory[chatHistory.length - 1].role !== 'assistant') {
            alert('Last message is not from assistant');
            return;
        }
        
        chatHistory.pop();
        var lastMsg = chatContainer.lastElementChild;
        if (lastMsg && lastMsg.classList.contains('assistant')) {
            lastMsg.parentNode.removeChild(lastMsg);
        }
        
        var provider = providerSelect ? providerSelect.value : 'groq';
        var model = modelSelect ? modelSelect.value : 'llama-3.1-70b-versatile';
        
        var loadingId = addMessage('Regenerating...', 'loading');
        disableInput(true);
        
        if (provider === 'puter') {
            callPuterAPI(chatHistory, model, function(response, error) {
                removeMessage(loadingId);
                if (error) {
                    addMessage('Error: ' + error, 'error');
                } else {
                    addMessage(response, 'assistant');
                    chatHistory.push({ role: 'assistant', content: response });
                    saveChatHistory();
                }
                disableInput(false);
            });
        } else {
            var payload = { messages: chatHistory, provider: provider, model: model };
            if (provider === 'custom' && customEndpoint && customKey) {
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
    }
    
    function callPuterAPI(messages, model, callback) {
        if (typeof puter === 'undefined' || !puter || !puter.ai || !puter.ai.chat) {
            callback(null, 'Puter AI not available. Use Puter browser for Puter models.');
            return;
        }
        
        var text = messages.map(function(m) { return m.role + ': ' + m.content; }).join('
');
        puter.ai.chat(text, { model: model })
            .then(function(response) {
                callback(response, null);
            })
            .catch(function(err) {
                callback(null, err.message || 'Puter API error');
            });
    }
    
    function clearChat() {
        if (!confirm('Clear all messages?')) return;
        if (chatContainer) chatContainer.innerHTML = '';
        chatHistory = [];
        saveChatHistory();
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
        if (sendBtn) sendBtn.textContent = disabled ? 'Sending...' : 'Send';
    }
    
    function saveChatHistory() {
        try {
            if (typeof localStorage !== 'undefined') {
                localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
            }
        } catch (e) {}
    }
    
    function loadChatHistory() {
        try {
            if (typeof localStorage !== 'undefined') {
                var stored = localStorage.getItem('chatHistory');
                if (stored) {
                    chatHistory = JSON.parse(stored);
                    for (var i = 0; i < chatHistory.length; i++) {
                        var msg = chatHistory[i];
                        addMessage(msg.content, msg.role === 'user' ? 'user' : 'assistant');
                    }
                }
            }
        } catch (e) {}
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
