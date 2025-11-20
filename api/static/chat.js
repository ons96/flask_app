(function() {
    'use strict';
    
    // DOM Elements
    var chatContainer = document.getElementById('chat-container');
    var userInput = document.getElementById('user-input');
    var sendBtn = document.getElementById('send-btn');
    var clearBtn = document.getElementById('clear-btn');
    var providerSelect = document.getElementById('provider-select');
    var modelInput = document.getElementById('model-input');
    var webSearchToggle = document.getElementById('web-search-toggle');
    var customEndpoint = document.getElementById('custom-endpoint');
    var customKey = document.getElementById('custom-key');
    var customApiFields = document.getElementById('custom-api-fields');
    var toggleSettings = document.getElementById('toggle-settings');
    var settingsPanel = document.getElementById('settings-panel');
    
    // State
    var chatHistory = [];
    var isPuterReady = false;
    
    // Initialize
    loadChatHistory();
    setupEventListeners();
    checkPuterStatus();
    
    // ========================================================================
    // EVENT LISTENERS
    // ========================================================================
    
    function setupEventListeners() {
        sendBtn.addEventListener('click', sendMessage);
        clearBtn.addEventListener('click', clearChat);
        toggleSettings.addEventListener('click', toggleSettingsPanel);
        
        userInput.addEventListener('keypress', function(e) {
            if (e.keyCode === 13 && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        providerSelect.addEventListener('change', function() {
            var provider = providerSelect.value;
            customApiFields.style.display = provider === 'custom' ? 'block' : 'none';
            
            // Update model placeholder based on provider
            if (provider === 'groq') {
                modelInput.value = 'llama-3.1-70b-versatile';
            } else if (provider === 'openrouter') {
                modelInput.value = 'meta-llama/llama-3.1-70b-instruct:free';
            } else if (provider === 'puter') {
                modelInput.value = 'claude-3.5-sonnet';
            }
        });
    }
    
    function toggleSettingsPanel() {
        if (settingsPanel.className.indexOf('open') > -1) {
            settingsPanel.className = 'settings-panel';
            toggleSettings.textContent = '⚙️ Settings';
        } else {
            settingsPanel.className = 'settings-panel open';
            toggleSettings.textContent = '▲ Hide Settings';
        }
    }
    
    // ========================================================================
    // CHAT FUNCTIONS
    // ========================================================================
    
    function sendMessage() {
        var message = userInput.value.trim();
        if (!message) return;
        
        var provider = providerSelect.value;
        var model = modelInput.value.trim();
        var webSearch = webSearchToggle.checked;
        
        // Add user message
        addMessage(message, 'user');
        chatHistory.push({ role: 'user', content: message });
        userInput.value = '';
        
        // Show loading
        var loadingId = addMessage('Thinking...', 'loading');
        disableInput(true);
        
        // Use Puter.js if selected
        if (provider === 'puter') {
            if (!isPuterReady) {
                removeMessage(loadingId);
                addMessage('Puter.js not ready. Please wait...', 'error');
                disableInput(false);
                return;
            }
            
            callPuterAI(model, message, function(response, error) {
                removeMessage(loadingId);
                if (error) {
                    addMessage('Puter Error: ' + error, 'error');
                } else {
                    addMessage(response, 'assistant', model, provider);
                    chatHistory.push({ role: 'assistant', content: response });
                    saveChatHistory();
                }
                disableInput(false);
            });
        } else {
            // Call backend API
            var payload = {
                messages: chatHistory,
                provider: provider,
                model: model,
                web_search: webSearch
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
                    addMessage(data.response, 'assistant', data.model, data.provider);
                    chatHistory.push({ role: 'assistant', content: data.response });
                    saveChatHistory();
                }
                
                disableInput(false);
            });
        }
    }
    
    function clearChat() {
        if (!confirm('Clear all messages?')) return;
        chatContainer.innerHTML = '';
        chatHistory = [];
        saveChatHistory();
    }
    
    // ========================================================================
    // UI FUNCTIONS
    // ========================================================================
    
    function addMessage(text, type, model, provider) {
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + type;
        messageDiv.id = 'msg-' + Date.now();
        
        var contentDiv = document.createElement('div');
        contentDiv.textContent = text;
        messageDiv.appendChild(contentDiv);
        
        if (model || provider) {
            var metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = (model || '') + (provider ? ' (' + provider + ')' : '');
            messageDiv.appendChild(metaDiv);
        }
        
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
    // STORAGE
    // ========================================================================
    
    function saveChatHistory() {
        try {
            localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        } catch (e) {
            console.log('LocalStorage error:', e);
        }
    }
    
    function loadChatHistory() {
        try {
            var stored = localStorage.getItem('chatHistory');
            if (stored) {
                chatHistory = JSON.parse(stored);
                // Restore messages to UI
                for (var i = 0; i < chatHistory.length; i++) {
                    var msg = chatHistory[i];
                    addMessage(msg.content, msg.role === 'user' ? 'user' : 'assistant');
                }
            }
        } catch (e) {
            console.log('LocalStorage error:', e);
        }
    }
    
    // ========================================================================
    // API CALLS
    // ========================================================================
    
    function callAPI(endpoint, payload, callback) {
        var xhr = new XMLHttpRequest();
        xhr.open('POST', endpoint, true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.timeout = 90000; // 90 seconds
        
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
        // Wait for Puter.js to load
        var checkInterval = setInterval(function() {
            if (typeof puter !== 'undefined' && puter.ai) {
                isPuterReady = true;
                console.log('Puter.js ready');
                clearInterval(checkInterval);
            }
        }, 100);
        
        // Timeout after 10 seconds
        setTimeout(function() {
            clearInterval(checkInterval);
            if (!isPuterReady) {
                console.log('Puter.js failed to load');
            }
        }, 10000);
    }
    
    function callPuterAI(model, message, callback) {
        if (!puter || !puter.ai) {
            callback(null, 'Puter.js not available');
            return;
        }
        
        try {
            puter.ai.chat(message, { model: model }).then(function(response) {
                callback(response, null);
            }).catch(function(error) {
                callback(null, error.message || 'Puter.js error');
            });
        } catch (e) {
            callback(null, e.message);
        }
    }
})();
