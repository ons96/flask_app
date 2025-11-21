// ES5 compatible version for older browsers like BlackBerry
(function () {
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
    window.addEventListener('load', function () {
        if (providerSelect) updateModelDropdown();
        setupEventListeners();
    });

    function setupEventListeners() {
        if (providerSelect) {
            providerSelect.addEventListener('change', updateModelDropdown);
        }

        if (fileBtn) {
            fileBtn.addEventListener('click', function (e) {
                e.preventDefault();
                showFileUpload();
            });
        }

        if (urlBtn) {
            urlBtn.addEventListener('click', function (e) {
                e.preventDefault();
                showUrlInput();
            });
        }

        if (jsonBtn) {
            jsonBtn.addEventListener('click', function (e) {
                e.preventDefault();
                showJsonInput();
            });
        }

        if (fileUpload) {
            fileUpload.addEventListener('change', loadCharacterFromFile);
        }

        if (loadBtn) {
            loadBtn.addEventListener('click', function (e) {
                e.preventDefault();
                loadCurrentInput();
            });
        }

        if (sendBtn) {
            sendBtn.addEventListener('click', function (e) {
                e.preventDefault();
                sendMessage();
            });
        }

        if (regenBtn) {
            regenBtn.addEventListener('click', function (e) {
                e.preventDefault();
                regenerateLastMessage();
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', function (e) {
                e.preventDefault();
                clearChat();
            });
        }

        if (userInput) {
            userInput.addEventListener('keypress', function (e) {
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

        // Clear existing options
        while (modelSelect.firstChild) {
            modelSelect.removeChild(modelSelect.firstChild);
        }

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

        // Reset file input so change event fires again for same file
        // e.target.value = ''; 

        if (file.name.toLowerCase().indexOf('.json') > -1) {
            var reader = new FileReader();
            reader.onload = function (evt) {
                try {
                    var character = JSON.parse(evt.target.result);
                    setCharacter(character);
                } catch (err) {
                    alert('Invalid JSON file: ' + err.message);
                }
            };
            reader.readAsText(file);
        } else if (file.name.toLowerCase().indexOf('.png') > -1) {
            // For PNG, we need to send it to the server to extract metadata
            // because doing it in ES5 JS on BlackBerry might be slow/complex
            uploadPNGCharacter(file);
        } else {
            alert('Please upload a .json or .png file');
        }
    }

    function uploadPNGCharacter(file) {
        addMessage('Uploading PNG character card...', 'loading');

        var formData = new FormData();
        formData.append('file', file);

        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/character/parse', true);

        xhr.onload = function () {
            if (xhr.status === 200) {
                try {
                    var data = JSON.parse(xhr.responseText);
                    if (data.success && data.character) {
                        setCharacter(data.character);
                    } else {
                        alert('Failed to parse character: ' + (data.error || 'Unknown error'));
                    }
                } catch (e) {
                    alert('Failed to parse response');
                }
            } else {
                alert('Upload failed: ' + xhr.status);
            }
            // Clear loading message
            var lastMsg = chatContainer.lastElementChild;
            if (lastMsg && lastMsg.className.indexOf('loading') !== -1) {
                chatContainer.removeChild(lastMsg);
            }
        };

        xhr.onerror = function () {
            alert('Network error during upload');
        };

        xhr.send(formData);
    }

    function loadCurrentInput() {
        if (loadMode === 'url' && urlInput) {
            loadCharacterFromURL(urlInput.value);
        } else if (loadMode === 'json' && jsonInput) {
            loadCharacterFromJSON(jsonInput.value);
        }
    }

    function loadCharacterFromURL(url) {
        if (!url || url.trim() === '') {
            alert('Please enter a URL');
            return;
        }
        url = url.trim();

        if (chatContainer) addMessage('Loading character from URL...', 'loading');

        var payload = { url: url };
        callAPI('/api/character/parse', payload, function (data, error) {
            // Clear loading message
            var lastMsg = chatContainer.lastElementChild;
            if (lastMsg && lastMsg.className.indexOf('loading') !== -1) {
                chatContainer.removeChild(lastMsg);
            }

            if (error) {
                alert('Failed to load character: ' + error);
            } else if (data && data.character) {
                setCharacter(data.character);
            }
        });
    }

    function loadCharacterFromJSON(jsonText) {
        if (!jsonText || jsonText.trim() === '') {
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
        if (userInput) {
            userInput.disabled = false;
            userInput.placeholder = 'Chat with ' + (character.name || (character.data && character.data.name) || 'Character') + '...';
        }
        if (sendBtn) sendBtn.disabled = false;
        if (regenBtn) regenBtn.disabled = false;

        conversationHistory = [];
        if (chatContainer) chatContainer.innerHTML = '';

        var charName = character.name || (character.data && character.data.name) || 'Character';
        addMessage('Character loaded: ' + charName, 'assistant');
        hideAllInputs();
    }

    function sendMessage() {
        if (!currentCharacter) {
            alert('Please load a character first');
            return;
        }

        if (!userInput) return;
        var message = userInput.value;
        if (!message || message.trim() === '') return;
        message = message.trim();

        var provider = providerSelect ? providerSelect.value : 'openrouter';
        var model = modelSelect ? modelSelect.value : 'meta-llama/llama-3.1-70b-instruct:free';

        addMessage(message, 'user');
        conversationHistory.push({ role: 'user', content: message });
        userInput.value = '';

        var loadingId = addMessage('Typing...', 'loading');
        disableInput(true);

        if (provider === 'puter') {
            callPuterRoleplayAPI(message, function (response, error) {
                handleResponse(response, error, loadingId);
            });
        } else {
            var payload = { message: message, character: currentCharacter, history: conversationHistory, provider: provider, model: model };
            callAPI('/api/roleplay/chat', payload, function (data, error) {
                if (data) {
                    handleResponse(data.response, error, loadingId);
                } else {
                    handleResponse(null, error, loadingId);
                }
            });
        }
    }

    function handleResponse(response, error, loadingId) {
        removeMessage(loadingId);
        if (error) {
            addMessage('Error: ' + error, 'error');
        } else {
            addMessage(response, 'assistant');
            conversationHistory.push({ role: 'assistant', content: response });
        }
        disableInput(false);
    }

    function regenerateLastMessage() {
        if (conversationHistory.length < 2 || conversationHistory[conversationHistory.length - 1].role !== 'assistant') {
            alert('No assistant message to regenerate');
            return;
        }

        conversationHistory.pop();
        var lastMsg = chatContainer.lastElementChild;
        if (lastMsg && lastMsg.className.indexOf('assistant') !== -1) {
            chatContainer.removeChild(lastMsg);
        }

        var lastUserMsg = conversationHistory[conversationHistory.length - 1].content;
        var provider = providerSelect ? providerSelect.value : 'openrouter';
        var model = modelSelect ? modelSelect.value : 'meta-llama/llama-3.1-70b-instruct:free';

        var loadingId = addMessage('Regenerating...', 'loading');
        disableInput(true);

        if (provider === 'puter') {
            callPuterRoleplayAPI(lastUserMsg, function (response, error) {
                handleResponse(response, error, loadingId);
            });
        } else {
            var payload = { message: lastUserMsg, character: currentCharacter, history: conversationHistory, provider: provider, model: model };
            callAPI('/api/roleplay/chat', payload, function (data, error) {
                if (data) {
                    handleResponse(data.response, error, loadingId);
                } else {
                    handleResponse(null, error, loadingId);
                }
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
        var fullPrompt = systemPrompt + '\n\nUser: ' + message + '\nAssistant:';

        puter.ai.chat(fullPrompt, { model: modelSelect.value })
            .then(function (response) {
                callback(response, null);
            })
            .catch(function (err) {
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
        messageDiv.id = 'msg-' + new Date().getTime();
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
        // Handle FormData (for file upload) vs JSON
        if (!(payload instanceof FormData)) {
            xhr.setRequestHeader('Content-Type', 'application/json');
        }

        xhr.timeout = 90000;

        xhr.onload = function () {
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

        xhr.onerror = function () {
            callback(null, 'Network error');
        };

        if (xhr.ontimeout) {
            xhr.ontimeout = function () {
                callback(null, 'Request timed out');
            };
        }

        try {
            if (payload instanceof FormData) {
                xhr.send(payload);
            } else {
                xhr.send(JSON.stringify(payload));
            }
        } catch (e) {
            callback(null, 'Failed to send request');
        }
    }
})();
