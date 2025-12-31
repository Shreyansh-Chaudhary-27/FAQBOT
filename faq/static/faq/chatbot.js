class GraphuraChatbot {
    constructor() {
        // State management
        this.state = {
            sessionId: null,
            userName: null,
            userEmail: null,
            userId: null,
            isInitialized: false,
            messages: [],
            isTyping: false,
            onboardingStep: 0, // 0: Start, 1: Waiting Name, 2: Waiting Email
            pendingQuery: null
        };

        // DOM Elements cache
        this.elements = {};

        // Initialize
        this.apiBase = '/api';
        this.init();
    }

    // Initialize the chatbot
    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadSession();
        this.showWelcomeMessage();

        console.log('✅ Graphura Chatbot initialized (Conversational Mode)');
    }

    // Cache DOM elements for performance
    cacheElements() {
        this.elements = {
            // Main containers
            chatbotContainer: document.getElementById('chatbotContainer'),
            chatMessages: document.getElementById('chatMessages'),
            chatInput: document.getElementById('chatInput'),
            sendBtn: document.getElementById('sendBtn'),
            chatbotToggle: document.getElementById('chatbotToggle'),

            // Actions
            minimizeBtn: document.getElementById('minimizeBtn'),
            maximizeBtn: document.getElementById('maximizeBtn'),
            closeBtn: document.getElementById('closeBtn'),
            resizeHandle: document.getElementById('resizeHandle'),

            // Indicators
            typingIndicator: document.getElementById('typingIndicator'),
            notificationBadge: document.getElementById('notificationBadge'),

            // Modal (We keep elements just in case, but won't use them for onboarding now)
            welcomeModal: document.getElementById('welcomeModal'),
            userForm: document.getElementById('userForm'),
            userNameInput: document.getElementById('userName'),
            userEmailInput: document.getElementById('userEmail'),

            // Quick buttons
            quickButtons: document.querySelectorAll('.quick-btn')
        };
    }

    // Bind event listeners
    bindEvents() {
        // Toggle chatbot
        if (this.elements.chatbotToggle) {
            this.elements.chatbotToggle.addEventListener('click', () => this.toggleChatbot());
        }

        // Send message
        if (this.elements.sendBtn) {
            this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        }

        if (this.elements.chatInput) {
            this.elements.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // Chat actions
        // Chat actions
        if (this.elements.minimizeBtn) this.elements.minimizeBtn.addEventListener('click', () => this.minimizeChat());
        if (this.elements.maximizeBtn) this.elements.maximizeBtn.addEventListener('click', () => this.toggleFullscreen());
        if (this.elements.closeBtn) this.elements.closeBtn.addEventListener('click', () => this.closeChat());

        if (this.elements.resizeHandle) this.initResize();

        // Quick buttons
        if (this.elements.quickButtons) {
            this.elements.quickButtons.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const question = e.currentTarget.dataset.question;
                    console.log('🔘 Quick button clicked:', question);
                    if (this.elements.chatInput && question) {
                        this.elements.chatInput.value = question;
                        this.sendMessage();
                    }
                });
            });
        }

        // Modal close (optional)
        if (this.elements.welcomeModal) {
            this.elements.welcomeModal.addEventListener('click', (e) => {
                if (e.target === this.elements.welcomeModal) {
                    this.closeModal();
                }
            });
        }
    }

    // Load session from localStorage
    loadSession() {
        try {
            const savedSession = localStorage.getItem('graphura_chat_session');
            if (savedSession) {
                const session = JSON.parse(savedSession);
                this.state = { ...this.state, ...session };

                // Restore messages
                if (this.state.messages && this.state.messages.length > 0) {
                    this.restoreMessages();
                }
            }
        } catch (error) {
            console.error('Error loading session:', error);
            this.clearSession();
        }
    }

    // Save session to localStorage
    saveSession() {
        const sessionData = {
            sessionId: this.state.sessionId,
            userName: this.state.userName,
            userEmail: this.state.userEmail,
            userId: this.state.userId,
            messages: this.state.messages.slice(-50), // Keep last 50 messages
            isInitialized: this.state.isInitialized
        };

        localStorage.setItem('graphura_chat_session', JSON.stringify(sessionData));
    }

    // Clear session
    clearSession() {
        localStorage.removeItem('graphura_chat_session');
        this.state = {
            sessionId: null,
            userName: null,
            userEmail: null,
            userId: null,
            isInitialized: false,
            messages: [],
            isTyping: false,
            onboardingStep: 0,
            pendingQuery: null
        };

        if (this.elements.chatMessages) this.elements.chatMessages.innerHTML = '';
        this.showWelcomeMessage();
    }

    // Show/hide chatbot
    toggleChatbot() {
        if (this.elements.chatbotContainer.classList.contains('active')) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        if (this.elements.chatbotContainer) this.elements.chatbotContainer.classList.add('active');
        if (this.elements.chatbotToggle) this.elements.chatbotToggle.innerHTML = '<i class="fas fa-times"></i>';
        if (this.elements.notificationBadge) {
            this.elements.notificationBadge.style.display = 'none';
        }

        // Focus input
        setTimeout(() => {
            if (this.elements.chatInput) this.elements.chatInput.focus();
        }, 400);

        this.scrollToBottom();
    }

    closeChat() {
        if (this.elements.chatbotContainer) this.elements.chatbotContainer.classList.remove('active');
        if (this.elements.chatbotToggle) this.elements.chatbotToggle.innerHTML = '<i class="fas fa-comment-dots"></i>';
    }

    toggleFullscreen() {
        if (!this.elements.chatbotContainer) return;

        // If minimized, expand first before going fullscreen
        if (this.elements.chatbotContainer.classList.contains('minimized')) {
            this.elements.chatbotContainer.classList.remove('minimized');
            if (this.elements.minimizeBtn) {
                this.elements.minimizeBtn.innerHTML = '<i class="fas fa-minus"></i>';
                this.elements.minimizeBtn.title = "Minimize";
            }
            // Restore dimensions if saved
            if (this._savedDimensions) {
                this.elements.chatbotContainer.style.width = this._savedDimensions.width;
                this.elements.chatbotContainer.style.height = this._savedDimensions.height;
            }

            // Small delay before going fullscreen for smooth transition
            setTimeout(() => {
                this.elements.chatbotContainer.classList.add('fullscreen');
                if (this.elements.maximizeBtn) {
                    this.elements.maximizeBtn.innerHTML = '<i class="fas fa-compress"></i>';
                    this.elements.maximizeBtn.title = "Restore";
                }
                setTimeout(() => this.scrollToBottom(), 350);
            }, 100);
            return;
        }

        this.elements.chatbotContainer.classList.toggle('fullscreen');
        const isFullscreen = this.elements.chatbotContainer.classList.contains('fullscreen');

        // Update icon
        if (this.elements.maximizeBtn) {
            this.elements.maximizeBtn.innerHTML = isFullscreen ? '<i class="fas fa-compress"></i>' : '<i class="fas fa-expand"></i>';
            this.elements.maximizeBtn.title = isFullscreen ? "Restore" : "Full Screen";
        }

        if (isFullscreen) {
            setTimeout(() => this.scrollToBottom(), 350);
        }
    }

    initResize() {
        const handle = this.elements.resizeHandle;
        const container = this.elements.chatbotContainer;
        if (!handle || !container) return;

        let startX, startY, startWidth, startHeight;

        const onMouseDown = (e) => {
            if (container.classList.contains('fullscreen')) return;
            e.preventDefault();
            startX = e.clientX;
            startY = e.clientY;
            startWidth = container.offsetWidth;
            startHeight = container.offsetHeight; // Use offsetHeight for border-box

            document.documentElement.addEventListener('mousemove', onMouseMove);
            document.documentElement.addEventListener('mouseup', onMouseUp);
            container.classList.add('resizing'); // Remove transition during drag
            container.style.transition = 'none';
        };

        const onMouseMove = (e) => {
            // Dragging Top-Left means moving Left increases width, moving Up increases height
            // Because anchored Bottom-Right
            const dx = startX - e.clientX;
            const dy = startY - e.clientY;

            container.style.width = Math.max(300, Math.min(window.innerWidth - 40, startWidth + dx)) + 'px';
            container.style.height = Math.max(400, Math.min(window.innerHeight - 40, startHeight + dy)) + 'px';
        };

        const onMouseUp = () => {
            document.documentElement.removeEventListener('mousemove', onMouseMove);
            document.documentElement.removeEventListener('mouseup', onMouseUp);
            container.classList.remove('resizing');
            // Restore transitions
            container.style.transition = '';
        };

        handle.addEventListener('mousedown', onMouseDown);
    }

    minimizeChat() {
        if (!this.elements.chatbotContainer) return;

        const isCurrentlyMinimized = this.elements.chatbotContainer.classList.contains('minimized');

        // If expanding from minimized state
        if (isCurrentlyMinimized) {
            // Remove minimized class to expand
            this.elements.chatbotContainer.classList.remove('minimized');

            // Restore saved dimensions if they exist
            if (this._savedDimensions) {
                this.elements.chatbotContainer.style.width = this._savedDimensions.width;
                this.elements.chatbotContainer.style.height = this._savedDimensions.height;
            }

            // Update button icon
            if (this.elements.minimizeBtn) {
                this.elements.minimizeBtn.innerHTML = '<i class="fas fa-minus"></i>';
                this.elements.minimizeBtn.title = "Minimize";
            }

            // Scroll to bottom after expand animation
            setTimeout(() => this.scrollToBottom(), 350);
        } else {
            // Collapsing: save current dimensions first
            this._savedDimensions = {
                width: this.elements.chatbotContainer.style.width || this.elements.chatbotContainer.offsetWidth + 'px',
                height: this.elements.chatbotContainer.style.height || this.elements.chatbotContainer.offsetHeight + 'px'
            };

            // Exit fullscreen if active before minimizing
            if (this.elements.chatbotContainer.classList.contains('fullscreen')) {
                this.elements.chatbotContainer.classList.remove('fullscreen');
                if (this.elements.maximizeBtn) {
                    this.elements.maximizeBtn.innerHTML = '<i class="fas fa-expand"></i>';
                    this.elements.maximizeBtn.title = "Full Screen";
                }
            }

            // Add minimized class to collapse
            this.elements.chatbotContainer.classList.add('minimized');

            // Update button icon
            if (this.elements.minimizeBtn) {
                this.elements.minimizeBtn.innerHTML = '<i class="fas fa-plus"></i>';
                this.elements.minimizeBtn.title = "Expand";
            }
        }
    }

    // Modal functions (kept for backward compatibility or reset)
    showModal() {
        if (this.elements.welcomeModal) {
            this.elements.welcomeModal.classList.add('active');
        }
    }
    closeModal() {
        if (this.elements.welcomeModal) {
            this.elements.welcomeModal.classList.remove('active');
        }
    }

    // Handle conversational onboarding
    async handleOnboardingFlow(message) {
        // Step 0 -> 1: User sent first message
        if (this.state.onboardingStep === 0) {
            this.state.pendingQuery = message; // Save original question
            this.state.onboardingStep = 1;

            this.showTyping(true);
            await new Promise(r => setTimeout(r, 600));
            this.showTyping(false);

            this.addMessage("Hello! 👋 Before I can help you with that, could you please tell me your name?", 'bot');
            return;
        }

        // Step 1 -> 2: Waiting for Name
        if (this.state.onboardingStep === 1) {
            if (message.length < 2) {
                this.addMessage("Please provide a valid name.", 'bot');
                return;
            }

            this.state.userName = message;
            this.state.onboardingStep = 2;

            this.showTyping(true);
            await new Promise(r => setTimeout(r, 600));
            this.showTyping(false);

            this.addMessage(`Nice to meet you, ${this.state.userName}! And what is your email address?`, 'bot');
            return;
        }

        // Step 2 -> 3: Waiting for Email & Finish
        if (this.state.onboardingStep === 2) {
            if (!this.validateEmail(message)) {
                this.addMessage("Please enter a valid email address.", 'bot');
                return;
            }

            this.state.userEmail = message;
            this.showTyping(true);

            let isNewUser = true;

            // Register user
            this.state.sessionId = 'session_' + Date.now();
            try {
                const res = await fetch(this.apiBase + '/users/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: this.state.userName,
                        email: this.state.userEmail,
                        session_id: this.state.sessionId
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    this.state.userId = data.id;
                    isNewUser = data.is_new;
                }
            } catch (error) {
                console.error("User registration failed", error);
            }

            this.state.isInitialized = true;
            this.state.onboardingStep = 3;
            this.saveSession();

            await new Promise(r => setTimeout(r, 600));
            this.showTyping(false);

            if (isNewUser) {
                this.addMessage("Thanks! Your details have been saved.", 'bot');
            } else {
                this.addMessage(`Welcome back, ${this.state.userName}!`, 'bot');
            }

            // Handle Pending Query
            const originalQuery = this.state.pendingQuery;
            const isGreeting = /^(hi|hello|hey|greetings|good morning|afternoon)/i.test(originalQuery);

            if (originalQuery && !isGreeting) {
                // Send the original query now
                this.showTyping(true);
                await this.sendToApi(originalQuery);
            } else {
                this.addMessage(`How can I help you today, ${this.state.userName}?`, 'bot');
            }
        }
    }

    // Send message function (Main Entry Point)
    async sendMessage() {
        if (!this.elements.chatInput) return;
        const message = this.elements.chatInput.value.trim();

        if (!message) return;

        // Visuals
        this.addMessage(message, 'user');
        this.elements.chatInput.value = '';

        // Check if initialized
        if (!this.state.isInitialized) {
            await this.handleOnboardingFlow(message);
            return;
        }

        // Normal Flow
        this.showTyping(true);
        await this.sendToApi(message);
    }

    // API Call helper
    async sendToApi(message) {
        try {
            console.log('📤 Sending message to API:', message);

            const response = await fetch(this.apiBase + '/chatbot/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    user_id: this.state.userId,
                    session_id: this.state.sessionId,
                    user_name: this.state.userName
                })
            });

            const data = await response.json();
            console.log('📥 API Response:', data);

            this.showTyping(false);

            if (data.success) {
                console.log('✅ Adding bot message:', data.response);
                this.addMessage(data.response, 'bot', data.confidence, data.source_faq_id);
            } else {
                const errorMsg = data.response || "I'm having trouble connecting to the server.";
                this.addMessage(errorMsg, 'bot');
            }

            this.saveSession();

        } catch (err) {
            console.error('Chat error:', err);
            this.showTyping(false);
            this.addMessage('Unable to reach assistant right now. Please try again.', 'bot');
        }
    }

    // Add message to chat
    addMessage(text, sender, confidence = null, faqId = null) {
        if (!this.elements.chatMessages) return;

        const messageId = 'msg_' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.id = messageId;

        let messageHTML = '';
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        if (sender === 'user') {
            // Get user initial for avatar
            const userInitial = this.state.userName ? this.state.userName.charAt(0).toUpperCase() : 'U';

            messageHTML = `
                <div class="message-content">
                    <div class="user-message-avatar">${userInitial}</div>
                    <div class="message user">
                        <div class="message-bubble">
                            <p>${this.escapeHtml(text)}</p>
                        </div>
                    </div>
                </div>
                <div class="message-time">${timestamp}</div>
            `;
            messageDiv.className = 'message-wrapper user';
        } else {
            // Bot message with avatar and feedback
            messageHTML = `
                <div class="message-content">
                    <div class="bot-message-avatar">
                        <span>G</span>
                    </div>
                    <div class="message bot">
                        <div class="message-bubble">
                            <p>${this.formatResponse(text)}</p>
                            <div class="message-footer">
                                <div class="feedback-buttons" data-message-id="${messageId}" data-faq-id="${faqId || ''}">
                                    <button class="feedback-btn thumbs-up" title="Helpful" onclick="chatbot.submitFeedback('${messageId}', 1, '${faqId || ''}')">
                                        <i class="fas fa-thumbs-up"></i>
                                    </button>
                                    <button class="feedback-btn thumbs-down" title="Not helpful" onclick="chatbot.submitFeedback('${messageId}', -1, '${faqId || ''}')">
                                        <i class="fas fa-thumbs-down"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="message-time">${timestamp}</div>
            `;
            messageDiv.className = 'message-wrapper bot';
        }

        messageDiv.innerHTML = messageHTML;
        this.elements.chatMessages.appendChild(messageDiv);

        this.state.messages.push({
            id: messageId,
            text: text,
            sender: sender,
            timestamp: new Date().toISOString(),
            confidence: confidence,
            faqId: faqId
        });

        this.scrollToBottom();
        return messageId;
    }

    // Submit feedback
    async submitFeedback(messageId, rating, faqId) {
        const feedbackContainer = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!feedbackContainer) return;

        const msgIndex = this.state.messages.findIndex(m => m.id === messageId);
        const userQuery = msgIndex > 0 ? this.state.messages[msgIndex - 1].text : '';
        const botResponse = this.state.messages[msgIndex]?.text || '';

        try {
            await fetch(this.apiBase + '/feedback/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message_id: messageId,
                    faq_id: faqId || null,
                    rating: rating,
                    query: userQuery,
                    response: botResponse,
                    session_id: this.state.sessionId,
                    user_email: this.state.userEmail
                })
            });

            feedbackContainer.innerHTML = `
                <span class="feedback-thanks ${rating > 0 ? 'positive' : 'negative'}">
                    <i class="fas ${rating > 0 ? 'fa-check-circle' : 'fa-times-circle'}"></i>
                    ${rating > 0 ? 'Thanks for the feedback!' : 'Sorry, we\'ll improve!'}
                </span>
            `;
            console.log('✅ Feedback submitted:', { messageId, rating, faqId });
        } catch (err) {
            console.error('Failed to submit feedback:', err);
            feedbackContainer.innerHTML = `<span class="feedback-error">Feedback failed</span>`;
        }
    }

    // Show welcome message
    showWelcomeMessage() {
        if (!this.elements.chatMessages) return;

        const welcomeHTML = `
            <div class="welcome-message">
                <h4 style="margin-bottom: 8px;">Welcome to Graphura Assistant!</h4>
                <p style="margin: 4px 0;">I can help you with:</p>
                <ul style="text-align: left; margin: 10px 0 12px; padding-left: 18px; line-height: 1.5;">
                    <li>Service information</li>
                    <li>Internship details</li>
                    <li>Technical support</li>
                </ul>
                <p style="margin: 0;">Click the chat button below to get started!</p>
            </div>
        `;
        this.elements.chatMessages.innerHTML = welcomeHTML;
    }

    restoreMessages() {
        if (!this.elements.chatMessages) return;
        this.elements.chatMessages.innerHTML = '';

        this.state.messages.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.id = msg.id;

            // Format stored timestamp or use current time
            const msgTime = msg.timestamp
                ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                : '';

            let messageHTML = '';
            if (msg.sender === 'user') {
                const userInitial = this.state.userName ? this.state.userName.charAt(0).toUpperCase() : 'U';
                messageHTML = `
                    <div class="message-content">
                        <div class="user-message-avatar">${userInitial}</div>
                        <div class="message user">
                            <div class="message-bubble">
                                <p>${this.escapeHtml(msg.text)}</p>
                            </div>
                        </div>
                    </div>
                    <div class="message-time">${msgTime}</div>`;
                messageDiv.className = 'message-wrapper user';
            } else {
                messageHTML = `
                    <div class="message-content">
                        <div class="bot-message-avatar">
                            <span>G</span>
                        </div>
                        <div class="message bot">
                            <div class="message-bubble">
                                <p>${this.formatResponse(msg.text)}</p>
                            </div>
                        </div>
                    </div>
                    <div class="message-time">${msgTime}</div>`;
                messageDiv.className = 'message-wrapper bot';
            }
            messageDiv.innerHTML = messageHTML;
            this.elements.chatMessages.appendChild(messageDiv);
        });
        this.scrollToBottom();
    }

    // Helpers
    scrollToBottom() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }
    }

    showTyping(show) {
        const existingTyping = document.getElementById('typingIndicatorInChat');

        if (show) {
            // Remove existing if any
            if (existingTyping) existingTyping.remove();

            // Create typing indicator in chat area
            const typingDiv = document.createElement('div');
            typingDiv.id = 'typingIndicatorInChat';
            typingDiv.className = 'message-wrapper bot typing-wrapper';
            typingDiv.innerHTML = `
                <div class="bot-message-avatar">
                    <span>G</span>
                </div>
                <div class="typing-indicator-chat">
                    <div class="typing-dots"><span></span><span></span><span></span></div>
                    <p>Graphura Assistant is typing...</p>
                </div>
            `;

            if (this.elements.chatMessages) {
                this.elements.chatMessages.appendChild(typingDiv);
            }
        } else {
            // Remove typing indicator
            if (existingTyping) existingTyping.remove();
        }

        this.state.isTyping = show;
        if (show) this.scrollToBottom();
    }

    validateEmail(email) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatResponse(text) {
        // Simple formatter
        let formatted = this.escapeHtml(text);
        formatted = formatted.replace(/\n/g, '<br>');
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>');
        return formatted;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new GraphuraChatbot();
});
