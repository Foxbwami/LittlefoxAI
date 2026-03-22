/* ===== AI INTERFACE FUNCTIONS (Index/Home) ===== */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

function initUnifiedSearchChatPage() {
    const input = document.getElementById('unifiedInput');
    const sendBtn = document.getElementById('unifiedSendBtn');
    const resultsEl = document.getElementById('results');
    const citationsEl = document.getElementById('citations');
    const badgeEl = document.getElementById('sourceBadge');
    const confidenceEl = document.getElementById('confidenceLabel');
    const conversationLog = document.getElementById('messages');
    const chatArea = document.getElementById('chatArea');
    const emptyState = document.getElementById('emptyState');
    const suggestionButtons = document.querySelectorAll('.suggestions button');
    const composer = document.getElementById('composer');
    const gptMain = document.getElementById('gptMain');
    const emptyComposerSlot = document.getElementById('emptyComposerSlot');
    const webToggle = document.getElementById('webToggle');
    const modePills = document.querySelectorAll('.mode-pills .pill');
    const sourcesPanel = document.getElementById('sourcesPanel');
    const newSessionBtn = document.getElementById('newSessionBtn');
    const attachBtn = document.getElementById('attachBtn');
    const fileInput = document.getElementById('fileInput');
    const attachmentHint = document.getElementById('attachmentHint');
    const typingIndicator = document.getElementById('typingIndicator');
    const convoList = document.getElementById('conversationList');
    const scrollBtn = document.getElementById('scrollToBottom');

    if (!input || !sendBtn || !conversationLog) return;

    let activeMode = 'chat';
    let lastQuery = '';
    let conversationId = localStorage.getItem('lfx_conversation_id') || Date.now().toString();
    let attachedFile = null;
    let conversations = JSON.parse(localStorage.getItem('lfx_conversations') || '[]');

    function setMode(mode) {
        activeMode = mode;
        modePills.forEach(pill => pill.classList.toggle('active', pill.dataset.mode === mode));
    }

    function startChat() {
        if (emptyState) emptyState.style.display = 'none';
        if (conversationLog) conversationLog.style.display = 'flex';
        if (chatArea) chatArea.classList.add('has-messages');
        if (composer && gptMain && composer.parentElement !== gptMain) {
            gptMain.appendChild(composer);
        }
        if (composer) composer.classList.remove('centered');
    }

    function resetChat() {
        if (emptyState) emptyState.style.display = 'block';
        if (conversationLog) conversationLog.style.display = 'none';
        if (chatArea) chatArea.classList.remove('has-messages');
        if (composer && emptyComposerSlot && composer.parentElement !== emptyComposerSlot) {
            emptyComposerSlot.appendChild(composer);
        }
        if (composer) composer.classList.add('centered');
    }

    function appendMessage(role, text, stream = false) {
        startChat();
        const wrapper = document.createElement('div');
        wrapper.className = `message ${role === 'user' ? 'user' : 'ai'}`;
        wrapper.textContent = stream ? '' : text;
        conversationLog.appendChild(wrapper);
        wrapper.scrollIntoView({ behavior: 'smooth' });

        if (stream) {
            typeText(wrapper, text);
        }
    }

    function typeText(element, text, speed = 14) {
        let index = 0;
        const interval = setInterval(() => {
            if (index < text.length) {
                element.textContent += text[index];
                index += 1;
                element.parentElement?.parentElement?.scrollIntoView({ behavior: 'smooth', block: 'end' });
            } else {
                clearInterval(interval);
            }
        }, speed);
    }

    function updateBadge(meta) {
        if (!badgeEl) return;
        const sourced = meta && meta.has_provenance;
        badgeEl.textContent = sourced ? 'Sourced' : 'Model';
        badgeEl.classList.toggle('sourced', sourced);
        badgeEl.classList.toggle('model', !sourced);
        if (confidenceEl) {
            const score = meta && typeof meta.source_confidence === 'number' ? meta.source_confidence : null;
            confidenceEl.textContent = score !== null ? `Confidence: ${score}` : 'Confidence: —';
        }
        if (sourcesPanel) {
            sourcesPanel.style.display = sourced ? 'grid' : 'none';
        }
    }

    function renderResults(results) {
        if (!resultsEl) return;
        resultsEl.innerHTML = '';
        const items = (results || []).slice(0, 5);
        if (sourcesPanel) {
            sourcesPanel.style.display = items.length ? 'grid' : 'none';
        }
        items.forEach((r, idx) => {
            const card = document.createElement('div');
            card.className = 'result';
            card.innerHTML = `
                <h4>${idx + 1}. ${escapeHtml(r.title || r.url || 'Result')}</h4>
                <small>${escapeHtml(r.url || '')}</small>
                <p>${escapeHtml(r.snippet || '')}</p>
                <div class="result-actions">
                    <button class="chip" onclick="openResult('${(r.url || '').replace(/'/g, '%27')}')">Open</button>
                    <button class="chip" onclick="previewResult('${(r.url || '').replace(/'/g, '%27')}')">Preview</button>
                    <button class="chip" onclick="askAIOnResult('summarize', '${(r.title || '').replace(/'/g, '%27')}', '${(r.snippet || '').replace(/'/g, '%27')}', '${(r.url || '').replace(/'/g, '%27')}')">Summarize</button>
                    <button class="chip" onclick="askAIOnResult('explain', '${(r.title || '').replace(/'/g, '%27')}', '${(r.snippet || '').replace(/'/g, '%27')}', '${(r.url || '').replace(/'/g, '%27')}')">Explain</button>
                </div>
            `;
            resultsEl.appendChild(card);
        });
    }

    function renderCitations(citations, sources) {
        if (!citationsEl) return;
        citationsEl.innerHTML = '';
        const used = sources || [];
        const list = citations && citations.length ? citations : used.map((s, i) => `[${i + 1}] ${s.title || s.url || 'Source'}`);
        citationsEl.innerHTML = list.map((label, i) => {
            const src = used[i] || {};
            const link = src.url || '#';
            const title = src.title || link;
            return `<div>${escapeHtml(label)}${link !== '#' ? ` <a href="${link}" target="_blank" rel="noopener noreferrer">${escapeHtml(title)}</a>` : ''}</div>`;
        }).join('');
    }

    async function submitQuery(query, forceSearch = false, isRegen = false) {
        if (!query) return;
        if (!isRegen) {
            appendMessage('user', query);
            input.value = '';
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 180) + 'px';
        }
        if (typingIndicator) typingIndicator.style.display = 'block';

        const wantsWeb = forceSearch || (webToggle && webToggle.checked);
        const endpoint = '/chat';
        const payload = {
            message: query,
            user_id: getUserId(),
            tone: activeMode === 'research' ? 'academic' : 'default',
            use_web: wantsWeb,
            conversation_id: conversationId,
            parent_query: lastQuery
        };
        if (attachedFile) {
            payload.message += ` [attached: ${attachedFile.name}]`;
        }

        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            const answer = data.answer || data.reply || 'No answer generated yet.';
            const usedSources = data.used_sources || data.sources || [];
            const results = data.results || usedSources;
            const citations = data.citations || [];
            const meta = {
                has_provenance: data.has_provenance || false,
                source_confidence: data.source_confidence,
            };

            updateBadge({
                has_provenance: data.has_provenance,
                source_confidence: data.source_confidence
            });
            renderResults(results);
            renderCitations(citations, usedSources);
            appendMessage('ai', answer, true);
            lastQuery = query;
            addHistory(query);
            if (!isRegen) saveConversationTurn(query, answer);
        } catch (err) {
            updateBadge({ has_provenance: false });
            console.error(err);
        } finally {
            if (typingIndicator) typingIndicator.style.display = 'none';
            attachedFile = null;
            if (attachmentHint) attachmentHint.style.display = 'none';
            if (fileInput) fileInput.value = '';
        }
    }

    sendBtn.addEventListener('click', () => submitQuery(input.value.trim()));
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            e.stopPropagation();
            submitQuery(input.value.trim());
        }
    });
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            e.stopPropagation();
        }
    });
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 180) + 'px';
    });

    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', () => {
            conversationId = Date.now().toString();
            localStorage.setItem('lfx_conversation_id', conversationId);
            conversationLog.innerHTML = '';
            lastQuery = '';
            renderResults([]);
            renderCitations([], []);
            updateBadge({ has_provenance: false });
            resetChat();
            input.focus();
        });
    }

    if (attachBtn && fileInput) {
        attachBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            attachedFile = file;
            if (attachmentHint) {
                attachmentHint.textContent = `Attached: ${file.name}`;
                attachmentHint.style.display = 'block';
            }
        });
    }

    modePills.forEach(pill => {
        pill.addEventListener('click', () => setMode(pill.dataset.mode || 'chat'));
    });

    if (scrollBtn) {
        scrollBtn.style.display = 'none';
    }

    function saveConversationTurn(userText, aiText) {
        let convo = conversations.find(c => c.id === conversationId);
        if (!convo) {
            convo = { id: conversationId, title: userText.slice(0, 40), messages: [] };
            conversations.unshift(convo);
        }
        convo.messages.push({ role: 'user', content: userText, ts: new Date().toISOString() });
        convo.messages.push({ role: 'ai', content: aiText, ts: new Date().toISOString() });
        localStorage.setItem('lfx_conversations', JSON.stringify(conversations.slice(0, 30)));
        renderConversationList();
    }

    function renderConversationList() {
        if (!convoList) return;
        convoList.innerHTML = '';
        const visible = conversations.filter(c => !c.archived);
        if (!visible.length) {
            convoList.innerHTML = '<div class="gpt-convo-item muted">No recent chats</div>';
            return;
        }
        const ordered = [
            ...visible.filter(c => c.pinned),
            ...visible.filter(c => !c.pinned)
        ];
        ordered.forEach(c => {
            const item = document.createElement('div');
            item.className = 'gpt-convo-item';
            item.innerHTML = `
                <span class="gpt-convo-title">${escapeHtml(c.title || 'New chat')}</span>
                <span class="gpt-convo-actions">
                    <button class="gpt-convo-btn kebab" data-menu title="More">⋯</button>
                    <div class="gpt-convo-menu" data-menu-panel>
                        <button data-share>Share</button>
                        <button data-group>Start a group chat</button>
                        <button data-rename>Rename</button>
                        <button data-move>Move to project</button>
                        <hr />
                        <button data-pin>${c.pinned ? 'Unpin chat' : 'Pin chat'}</button>
                        <button data-archive>Archive</button>
                        <button class="danger" data-delete>Delete</button>
                    </div>
                </span>
            `;
            item.addEventListener('click', (e) => {
                if (e.target.closest('[data-menu]') || e.target.closest('[data-menu-panel]')) return;
                loadConversation(c.id);
            });
            const menuBtn = item.querySelector('[data-menu]');
            const menuPanel = item.querySelector('[data-menu-panel]');
            if (menuBtn && menuPanel) {
                menuBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    document.querySelectorAll('.gpt-convo-menu').forEach(m => {
                        if (m !== menuPanel) m.classList.remove('open');
                    });
                    menuPanel.classList.toggle('open');
                });
            }

            item.querySelector('[data-delete]')?.addEventListener('click', (e) => {
                e.stopPropagation();
                conversations = conversations.filter(x => x.id !== c.id);
                localStorage.setItem('lfx_conversations', JSON.stringify(conversations.slice(0, 30)));
                if (conversationId === c.id) {
                    conversationId = Date.now().toString();
                    localStorage.setItem('lfx_conversation_id', conversationId);
                    conversationLog.innerHTML = '';
                    resetChat();
                }
                renderConversationList();
                if (window.showToast) showToast('Conversation deleted');
            });
            item.querySelector('[data-share]')?.addEventListener('click', async (e) => {
                e.stopPropagation();
                const shareId = `lfx_share_${c.id}`;
                try {
                    localStorage.setItem(shareId, JSON.stringify(c));
                } catch (err) {}
                const link = `${window.location.origin}/?share=${encodeURIComponent(c.id)}`;
                try {
                    await navigator.clipboard.writeText(link);
                    if (window.showToast) showToast('Share link copied');
                } catch (err) {
                    if (window.showToast) showToast('Unable to copy link');
                }
            });
            item.querySelector('[data-rename]')?.addEventListener('click', (e) => {
                e.stopPropagation();
                const name = prompt('Rename chat', c.title || '');
                if (name !== null) {
                    c.title = name.trim() || c.title || 'New chat';
                    localStorage.setItem('lfx_conversations', JSON.stringify(conversations.slice(0, 30)));
                    renderConversationList();
                }
            });
            item.querySelector('[data-pin]')?.addEventListener('click', (e) => {
                e.stopPropagation();
                c.pinned = !c.pinned;
                localStorage.setItem('lfx_conversations', JSON.stringify(conversations.slice(0, 30)));
                renderConversationList();
            });
            item.querySelector('[data-archive]')?.addEventListener('click', (e) => {
                e.stopPropagation();
                c.archived = true;
                localStorage.setItem('lfx_conversations', JSON.stringify(conversations.slice(0, 30)));
                renderConversationList();
                if (window.showToast) showToast('Chat archived');
            });
            item.querySelector('[data-group]')?.addEventListener('click', (e) => {
                e.stopPropagation();
                if (window.showToast) showToast('Group chat coming soon');
            });
            item.querySelector('[data-move]')?.addEventListener('click', (e) => {
                e.stopPropagation();
                if (window.showToast) showToast('Move to project coming soon');
            });
            convoList.appendChild(item);
        });
    }

    function loadConversation(id) {
        const convo = conversations.find(c => c.id === id);
        if (!convo) return;
        conversationId = id;
        localStorage.setItem('lfx_conversation_id', conversationId);
        conversationLog.innerHTML = '';
        startChat();
        convo.messages.forEach(m => {
            appendMessage(m.role === 'user' ? 'user' : 'ai', m.content, false);
        });
    }

    // Load shared conversation if present
    const params = new URLSearchParams(window.location.search);
    const shareParam = params.get('share');
    if (shareParam) {
        const shared = localStorage.getItem(`lfx_share_${shareParam}`);
        if (shared) {
            try {
                const convo = JSON.parse(shared);
                conversationLog.innerHTML = '';
                startChat();
                (convo.messages || []).forEach(m => {
                    appendMessage(m.role === 'user' ? 'user' : 'ai', m.content, false);
                });
            } catch (err) {}
        }
    }

    renderConversationList();

    setMode('chat');
    updateBadge({ has_provenance: false });
    if (conversations.length === 0) {
        resetChat();
    } else if (conversationLog && conversationLog.children.length === 0) {
        resetChat();
    }

    suggestionButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const text = btn.getAttribute('data-suggest') || btn.textContent;
            input.value = text;
            input.focus();
        });
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.gpt-convo-actions')) {
            document.querySelectorAll('.gpt-convo-menu').forEach(m => m.classList.remove('open'));
        }
    });
}
function initAIInterface() {
    const textarea = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatMessages = document.getElementById('chatMessages');
    
    if (!textarea || !sendBtn || !chatMessages) return;

    // Auto-resize textarea
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    });

    // Send message to appropriate endpoint based on mode
    async function sendAIMessage() {
        const message = textarea.value.trim();
        if (!message) return;

        // Display user message
        addAIMessage('user', message);
        textarea.value = '';
        textarea.style.height = 'auto';

        const mode = document.querySelector('.ai-mode-btn.active')?.dataset.mode || 'chat';
        
        try {
            // Determine endpoint based on mode
            let endpoint = '/chat';
            let payload = { message, user_id: getUserId() };

            if (['research', 'code'].includes(mode)) {
                endpoint = '/search';
                payload = { query: message, user_id: getUserId(), tone: mode === 'research' ? 'academic' : 'default' };
            } else if (mode === 'analyze') {
                // Handle file analysis
                addAIMessage('ai', 'File analysis mode activated. Please attach a file to analyze.');
                return;
            } else if (['image', 'video', 'music', 'translate'].includes(mode)) {
                // These modes would require additional backend endpoints
                addAIMessage('ai', `${mode} mode is coming soon. For now, use Chat mode.`);
                return;
            } else {
                // Default chat mode
                payload.tone = 'default';
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            
            if (endpoint === '/search') {
                const answer = data.answer || 'No answer generated.';
                addAIMessage('ai', answer);
                if (data.results && data.results.length > 0) {
                    addAIMessage('ai', `Found ${data.results.length} sources. Type "show sources" to see them.`);
                }
            } else {
                const reply = data.reply || 'I couldn\'t process that request.';
                addAIMessage('ai', reply);
            }

            addHistory(message);
        } catch (error) {
            addAIMessage('ai', 'Error: Unable to process your request. Please try again.');
            console.error(error);
        }
    }

    // Send on button click
    sendBtn.addEventListener('click', sendAIMessage);

    // Send on Ctrl+Enter
    textarea.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            sendAIMessage();
        }
    });

    // Focus on load
    window.addEventListener('load', () => textarea.focus());
}

function addAIMessage(role, text) {
    const container = document.getElementById('chatMessages');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.className = `ai-message-wrapper ${role}`;

    const content = document.createElement('div');
    content.className = 'ai-message-content';
    content.innerHTML = `<strong>${role === 'user' ? 'You' : 'AI'}:</strong> ${text}`;

    wrapper.appendChild(content);
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAIInterface);
} else {
    initAIInterface();
}

/* ===== EXISTING FUNCTIONS ===== */
function getUserId() {
    let id = localStorage.getItem("user_id");
    if (!id) {
        id = "user_" + Math.random().toString(36).slice(2);
        localStorage.setItem("user_id", id);
    }
    return id;
}

function addHistory(query) {
    const key = "search_history";
    const items = JSON.parse(localStorage.getItem(key) || "[]");
    items.unshift({ q: query, t: new Date().toISOString() });
    localStorage.setItem(key, JSON.stringify(items.slice(0, 20)));
}

function renderHistory() {
    const list = document.getElementById("historyList");
    if (!list) return;
    const items = JSON.parse(localStorage.getItem("search_history") || "[]");
    list.innerHTML = items.map(i => `<li>${i.q}</li>`).join("");
}

async function runSearch(query) {
    const answerEl = document.getElementById("answer");
    const resultsEl = document.getElementById("results");
    const citationsEl = document.getElementById("citations");
    const academicToggle = document.getElementById("academicMode");
    const citationStyle = document.getElementById("citationStyle");
    const strictSources = document.getElementById("strictSources");
    const academicTemplate = document.getElementById("academicTemplate");
    const academicInstructions = document.getElementById("academicInstructions");
    const bibtexEl = document.getElementById("bibtex");
    const endnoteEl = document.getElementById("endnote");
    if (!answerEl || !resultsEl) return;

    answerEl.textContent = "Searching...";
    resultsEl.innerHTML = "";
    if (citationsEl) citationsEl.innerHTML = "";
    if (bibtexEl) bibtexEl.textContent = "";
    if (endnoteEl) endnoteEl.textContent = "";

    const res = await fetch("/search", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            query,
            user_id: getUserId(),
            tone: academicToggle && academicToggle.checked ? "academic" : "default",
            citation_style: citationStyle ? citationStyle.value : "APA",
            strict_sources: strictSources ? strictSources.checked : false,
            academic_template: academicTemplate ? academicTemplate.value : "summary",
            academic_instructions: academicInstructions ? academicInstructions.value.trim() : ""
        })
    });
    const data = await res.json();

    answerEl.textContent = data.answer || "No answer generated yet.";
    const shownResults = (data.results || []).slice(0, 5);
    // Clear and build result cards (interactive actions)
    resultsEl.innerHTML = '';
    shownResults.forEach((r, idx) => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const title = document.createElement('h3');
        title.textContent = `${idx + 1}. ${r.title || 'Result'}`;
        card.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'muted';
        meta.textContent = r.source || '';
        card.appendChild(meta);

        const urlSmall = document.createElement('small');
        urlSmall.textContent = r.url || '';
        card.appendChild(urlSmall);

        const para = document.createElement('p');
        para.innerHTML = r.snippet || '';
        card.appendChild(para);

        const actions = document.createElement('div');
        actions.className = 'result-actions';

        const openBtn = document.createElement('button');
        openBtn.className = 'chip';
        openBtn.textContent = 'Open';
        openBtn.addEventListener('click', () => openResult(r.url));
        actions.appendChild(openBtn);

        const previewBtn = document.createElement('button');
        previewBtn.className = 'chip';
        previewBtn.textContent = 'Preview';
        previewBtn.addEventListener('click', () => previewResult(r.url));
        actions.appendChild(previewBtn);

        const sumBtn = document.createElement('button');
        sumBtn.className = 'chip';
        sumBtn.textContent = 'Summarize';
        sumBtn.addEventListener('click', () => askAIOnResult('summarize', r.title, r.snippet, r.url));
        actions.appendChild(sumBtn);

        const explainBtn = document.createElement('button');
        explainBtn.className = 'chip';
        explainBtn.textContent = 'Explain';
        explainBtn.addEventListener('click', () => askAIOnResult('explain', r.title, r.snippet, r.url));
        actions.appendChild(explainBtn);

        card.appendChild(actions);
        resultsEl.appendChild(card);
    });
    if (citationsEl) {
        const citationList = data.citations || shownResults.map((r, i) => `[${i + 1}] ${r.title || r.url || "Source"}`);
        const citationSources = data.used_sources || shownResults;
        citationsEl.innerHTML = citationList.map((label, i) => {
            const src = citationSources[i] || {};
            const link = src.url || "#";
            return `<div>${label}${link !== "#" ? ` <a href="${link}" target="_blank" rel="noopener noreferrer">${src.title || link}</a>` : ""}</div>`;
        }).join("");
    }
    if (bibtexEl && data.bibtex) {
        bibtexEl.textContent = data.bibtex;
    }
    if (endnoteEl && data.endnote) {
        endnoteEl.textContent = data.endnote;
    }
    addHistory(query);
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([answerEl]).catch(() => {});
    }
}

// Open result in a new tab (acts like browser)
function openResult(url) {
    if (!url) return;
    window.open(url, '_blank');
}

// Preview result in an in-page modal
function previewResult(url) {
    if (!url) return;
    const modal = document.getElementById('previewModal');
    const frame = document.getElementById('previewFrame');
    if (!modal || !frame) return;
    frame.src = url;
    modal.style.display = 'flex';
}

function closePreview() {
    const modal = document.getElementById('previewModal');
    const frame = document.getElementById('previewFrame');
    if (!modal || !frame) return;
    frame.src = 'about:blank';
    modal.style.display = 'none';
}

// Ask AI to operate on a chosen result (summarize/explain/etc.)
async function askAIOnResult(action, title, snippet, url) {
    const insightsList = document.getElementById('insightsList');
    if (insightsList) insightsList.innerHTML = `<li>Requesting ${action} for selected result...</li>`;
    try {
        const payload = {
            query: `${action.toUpperCase()}: ${title || url}\n\n${snippet || ''}`,
            action,
            url,
            user_id: getUserId()
        };
        const res = await fetch('/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        const summary = data.answer || data.summary || 'No summary returned.';
        if (insightsList) insightsList.innerHTML = `<li><strong>${action}:</strong> ${summary}</li>`;
    } catch (err) {
        if (insightsList) insightsList.innerHTML = `<li style="color: #ff6b6b">Error: Unable to request AI action</li>`;
        console.error(err);
    }
}

function initSearchPage() {
    const input = document.getElementById("q");
    const btn = document.getElementById("searchBtn");
    if (!input || !btn) return;
    btn.addEventListener("click", () => {
        const q = input.value.trim();
        if (q) runSearch(q);
    });
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            const q = input.value.trim();
            if (q) runSearch(q);
        }
    });
}

// New init for updated search.html
function initEnhancedSearchPage() {
    const input = document.getElementById('searchInput');
    const btn = document.getElementById('searchBtn');
    const tabs = document.querySelectorAll('.results-tabs .tab');
    const insightsList = document.getElementById('insightsList');
    const resultsSection = document.getElementById('resultsSection');

    if (!input || !btn || !resultsSection) return;

    btn.addEventListener('click', () => {
        const q = input.value.trim();
        if (q) runSearch(q);
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const q = input.value.trim();
            if (q) runSearch(q);
        }
    });

    tabs.forEach(tab => tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        // In a complete implementation we'd refetch or filter results here.
    }));

    document.getElementById('generateReportBtn')?.addEventListener('click', () => {
        const snippets = Array.from(document.querySelectorAll('.result-card p')).slice(0,5).map(p => p.textContent).join('\n\n');
        if (!snippets) return showToast('No results to summarize');
        // Simulate a summary generation call
        insightsList.innerHTML = `<li>Generating summary...</li>`;
        setTimeout(() => {
            insightsList.innerHTML = `<li>Top trends: 1) Example trend A 2) Example trend B</li><li>Key finding: Sample finding derived from top results.</li>`;
        }, 800);
    });

    // Prefill and focus
    const params = new URLSearchParams(window.location.search);
    const preq = params.get('q');
    if (preq) { input.value = preq; runSearch(preq); }
    input.focus();
}

// Wire multipurpose mode buttons and advanced panel
document.addEventListener('DOMContentLoaded', () => {
    const modeBtns = document.querySelectorAll('.mode-btn');
    modeBtns.forEach(b => b.addEventListener('click', () => {
        modeBtns.forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        const mode = b.dataset.mode || 'multipurpose';
        const input = document.getElementById('searchInput');
        if (!input) return;
        const placeholders = {
            multipurpose: 'Ask, search, summarize, code, or analyze...',
            research: 'Search academic papers, datasets, and reports...',
            code: 'Search code snippets, documentation, or ask for fixes...',
            creative: 'Search for ideas, prompts, or creative inspiration...'
        };
        input.placeholder = placeholders[mode] || placeholders.multipurpose;
    }));

    const advToggle = document.getElementById('advancedToggle');
    const advPanel = document.getElementById('advancedPanel');
    if (advToggle && advPanel) {
        advToggle.addEventListener('click', () => {
            const showing = advPanel.style.display !== 'none';
            advPanel.style.display = showing ? 'none' : 'block';
            advToggle.textContent = showing ? 'Advanced ▾' : 'Advanced ▴';
        });
    }
});

// Hero chips and Ask AI wiring
document.addEventListener('DOMContentLoaded', () => {
    const chips = document.querySelectorAll('.suggestion-chips .chip');
    const searchInput = document.getElementById('searchInput');
    const askBtn = document.getElementById('askAiBtn');
    const searchBtn = document.getElementById('searchBtn');

    chips.forEach(c => c.addEventListener('click', () => {
        if (!searchInput) return;
        searchInput.value = c.textContent;
        runSearch(c.textContent);
    }));

    if (askBtn && searchInput) {
        askBtn.addEventListener('click', () => {
            const q = searchInput.value.trim();
            if (!q) return;
            // Ask AI: run search and show the answer in assistant panel (insights)
            runSearch(q);
            // Minor UI feedback: focus
            searchInput.focus();
        });
    }

    if (searchBtn && searchInput) {
        searchBtn.addEventListener('click', () => {
            const q = searchInput.value.trim();
            if (!q) return;
            runSearch(q);
        });
    }
});

function addActionBar(messageEl, text, isAi = false) {
    if (!isAi) return;
    const actionBar = document.createElement("div");
    actionBar.className = "message-actions";
    actionBar.innerHTML = `
        <button class="action-btn" title="Copy" onclick="copyToClipboard('${btoa(text).replace(/"/g, '&quot;')}')">
            <i class="fas fa-copy"></i>
        </button>
        <button class="action-btn" title="Copy as Markdown" onclick="copyAsMarkdown('${btoa(text).replace(/"/g, '&quot;')}')">
            <i class="fas fa-markdown"></i>
        </button>
        <button class="action-btn" title="Save to library" onclick="saveToLibrary('${btoa(text).replace(/"/g, '&quot;')}')">
            <i class="fas fa-bookmark"></i>
        </button>
        <button class="action-btn" title="Share" onclick="shareResponse('${btoa(text).replace(/"/g, '&quot;')}')">
            <i class="fas fa-share-alt"></i>
        </button>
    `;
    messageEl.appendChild(actionBar);
}

function addMessage(role, text, isTyped = false) {
    const chat = document.getElementById("chat");
    if (!chat) return;
    const wrapper = document.createElement("div");
    wrapper.className = role === "You" ? "message from-user" : "message from-ai";
    const contentDiv = document.createElement("div");
    contentDiv.innerHTML = `<div class="role">${role}</div><div class="message-content"></div>`;
    wrapper.appendChild(contentDiv);
    
    if (isTyped && role === "AI") {
        const msgContent = contentDiv.querySelector(".message-content");
        typeWriterEffect(msgContent, text);
        addActionBar(wrapper, text, true);
    } else {
        contentDiv.querySelector(".message-content").textContent = text;
        addActionBar(wrapper, text, role === "AI");
    }
    
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([wrapper]).catch(() => {});
    }
}

function typeWriterEffect(element, text, speed = 20) {
    element.innerHTML = '';
    let index = 0;
    const interval = setInterval(() => {
        if (index < text.length) {
            element.textContent += text[index];
            index++;
            element.parentElement.parentElement.scrollIntoView({ behavior: 'smooth' });
        } else {
            clearInterval(interval);
        }
    }, speed);
}

function initChatPage() {
    const msgInput = document.getElementById("msg");
    const sendBtn = document.getElementById("sendBtn");
    if (!msgInput || !sendBtn) return;
    autoResize(msgInput);
    msgInput.addEventListener("input", () => autoResize(msgInput));
    sendBtn.addEventListener("click", () => sendChat(msgInput));
    msgInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendChat(msgInput);
        }
    });
}

async function sendChat(msgInput) {
    const msg = msgInput.value.trim();
    if (!msg) return;
    addMessage("You", msg);
    msgInput.value = "";
    autoResize(msgInput);
    const academicToggle = document.getElementById("academicModeChat");
    const academicInstructionsChat = document.getElementById("academicInstructionsChat");
    const lower = msg.toLowerCase();
    const execute = (msg.includes("```") && (lower.includes("run") || lower.includes("execute"))) || lower.startsWith("/run ");

    const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            message: msg,
            user_id: getUserId(),
            tone: academicToggle && academicToggle.checked ? "academic" : "default",
            academic_instructions: academicInstructionsChat ? academicInstructionsChat.value.trim() : "",
            execute
        })
    });
    const data = await res.json();
    addMessage("AI", data.reply || "", true);
}

function autoResize(textarea) {
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 220) + "px";
}

function initProfilePage() {
    const nameEl = document.getElementById("name");
    const personalityEl = document.getElementById("personality");
    const saveBtn = document.getElementById("saveProfile");
    if (!nameEl || !personalityEl || !saveBtn) return;

    fetch(`/profile?user_id=${encodeURIComponent(getUserId())}`)
        .then(res => res.json())
        .then(data => {
            nameEl.value = data.name || "";
            personalityEl.value = data.personality || "";
        });

    saveBtn.addEventListener("click", () => {
        fetch("/profile", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                user_id: getUserId(),
                name: nameEl.value.trim(),
                personality: personalityEl.value.trim()
            })
        }).then(() => {
            const status = document.getElementById("profileStatus");
            if (status) {
                status.textContent = "Saved";
                setTimeout(() => { status.textContent = ""; }, 2000);
            }
        });
    });
}

function initFeedbackPage() {
    const form = document.getElementById("feedbackForm");
    if (!form) return;
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = document.getElementById("feedbackText").value.trim();
        const rating = document.getElementById("rating").value;
        if (!text) return;
        await fetch("/feedback", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ response: text, rating })
        });
        document.getElementById("feedbackText").value = "";
        const status = document.getElementById("feedbackStatus");
        if (status) status.textContent = "Thanks for the feedback!";
    });
}

renderHistory();
initUnifiedSearchChatPage();
initSearchPage();
initChatPage();
initProfilePage();
initFeedbackPage();
/* Global Keyboard Shortcuts */
document.addEventListener('keydown', function(e) {
    // ? = Show help
    if (e.key === '?' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
        showHelpModal();
    }
    
    // Ctrl/Cmd + K = Command palette
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        showCommandPalette();
    }
    
    // Ctrl/Cmd + L = New chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        window.location.href = '/chat-ui';
    }
    
    // Ctrl/Cmd + S = Save
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        const lastMessage = document.querySelector('.message.from-ai:last-of-type');
        if (lastMessage) {
            const text = lastMessage.textContent || lastMessage.innerText;
            saveToLibrary(btoa(text));
        }
    }
    
    // Ctrl/Cmd + E = Export
    if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        const lastMessage = document.querySelector('.message.from-ai:last-of-type');
        if (lastMessage) {
            const text = lastMessage.textContent || lastMessage.innerText;
            exportResponse(btoa(text));
        }
    }
});

/* Action Functions */
function copyToClipboard(encodedText) {
    const text = atob(encodedText);
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!');
    });
}

function copyAsMarkdown(encodedText) {
    const text = atob(encodedText);
    const markdown = '> ' + text.split('\\n').join('\\n> ');
    navigator.clipboard.writeText(markdown).then(() => {
        showToast('Copied as markdown!');
    });
}

function saveToLibrary(encodedText) {
    const text = atob(encodedText);
    const saved = JSON.parse(localStorage.getItem('saved_responses') || '[]');
    saved.push({
        id: Date.now(),
        text: text,
        savedAt: new Date().toISOString(),
        userId: getUserId()
    });
    localStorage.setItem('saved_responses', JSON.stringify(saved.slice(-50)));
    showToast('Saved to library!');
}

function shareResponse(encodedText) {
    const text = atob(encodedText);
    const shareUrl = `${window.location.origin}?shared=${btoa(text)}`;
    navigator.clipboard.writeText(shareUrl).then(() => {
        showToast('Share link copied!');
    });
}

function exportResponse(encodedText) {
    const text = atob(encodedText);
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', `response_${Date.now()}.txt`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    showToast('Exported as .txt!');
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: rgba(64, 201, 162, 0.2);
        border: 1px solid rgba(64, 201, 162, 0.5);
        color: var(--accent, #40c9a2);
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 13px;
        z-index: 10000;
        animation: slideUp 0.3s ease;
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => document.body.removeChild(toast), 300);
    }, 3000);
}

function showHelpModal() {
    alert(`Keyboard Shortcuts:
    ? - Show this help
    Ctrl/Cmd + K - Command palette
    Ctrl/Cmd + L - New chat
    Ctrl/Cmd + S - Save response  
    Ctrl/Cmd + E - Export response`);
}

function showCommandPalette() {
    const command = prompt('Command palette (type command):');
    if (!command) return;
    const cmd = command.toLowerCase().trim();
    if (cmd.includes('search') || cmd.includes('explore')) {
        window.location.href = '/explore';
    } else if (cmd.includes('chat')) {
        window.location.href = '/chat-ui';
    } else if (cmd.includes('profile')) {
        window.location.href = '/profile-ui';
    } else if (cmd.includes('history')) {
        window.location.href = '/history';
    } else if (cmd.includes('clear')) {
        if (confirm('Clear chat history?')) {
            localStorage.removeItem('search_history');
            location.reload();
        }
    }
}
