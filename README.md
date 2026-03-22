# LittlefoxAI - Comprehensive System Documentation

**Last Updated:** March 22, 2026

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Backend Components](#backend-components)
4. [Frontend Components](#frontend-components)
5. [Features](#features)
6. [API Endpoints](#api-endpoints)
7. [File Structure](#file-structure)
8. [Data Flow](#data-flow)
9. [Technology Stack](#technology-stack)
10. [Getting Started](#getting-started)

---

## System Overview

**LittlefoxAI** is a hybrid AI system that combines semantic memory, distributed crawling, and advanced search capabilities. It serves as an intelligent assistant with multiple interaction modes optimized for different use cases.

### Core Philosophy
- **User-Centric Design**: AI-first interface with intuitive mode switching
- **Semantic Understanding**: Context-aware conversations using vector embeddings
- **Hybrid Search**: Combines local knowledge base with live web search
- **Privacy-First**: Conversation history stored locally in browser
- **Personalization**: Customizable personality and communication style

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (UI/UX)                        │
│  ┌──────────────┬──────────────┬──────────────┬────────────┐ │
│  │ Index.html   │ Chat.html    │ Search.html  │ Info Pages │ │
│  │ (Main AI)    │ (Dedicated)  │ (Research)   │ (Profile)  │ │
│  └──────────────┴──────────────┴──────────────┴────────────┘ │
│         ↓           ↓                ↓              ↓           │
│  ┌────────────────────────────────────────────────────────┐   │
│  │        Static Assets (CSS, JavaScript)                │   │
│  │  • style.css (2070 lines) - Unified AI styling        │   │
│  │  • app.js (521 lines) - Shared functionality          │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────┐
         │    Flask Backend (Python)            │
         │    Port: 5000 (typically)            │
         └──────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │         Backend Core Components           │
        ├───────────────────────────────────────────┤
        │ • GPTMini Model + Tokenizer             │
        │ • Vector Store (FAISS)                  │
        │ • Search Index                          │
        │ • Chat Memory Management                │
        │ • Web Crawler & Scheduler               │
        └───────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │           Data & Storage Layer            │
        ├───────────────────────────────────────────┤
        │ • Local Knowledge Base (learning_data)  │
        │ • Embeddings Cache (FAISS index)        │
        │ • User Profiles & Settings              │
        │ • Processed Data & Tokenizer            │
        │ • Model Weights (model.pth)             │
        └───────────────────────────────────────────┘
```

---

## Backend Components

### 1. **Core AI Model** (`model.py`, `train.py`, `fine_tune.py`)
- **Model**: GPTMini - lightweight transformer-based language model
- **Purpose**: Generate contextual responses for chat and search
- **Tokenizer**: BPE (Byte-Pair Encoding) with customizable vocabulary
- **Training**: Supervised fine-tuning on curated learning data
- **Inference**: Real-time response generation with temperature control

**Key Files**:
- `model.py` - Model architecture and forward pass
- `train.py` - Training pipeline with loss calculation
- `finetune.py` - Fine-tuning on custom datasets
- `tokenizer.py` - Tokenization logic
- `tokenizer_bpe.py` - BPE tokenizer implementation
- `saved_model/model.pth` - Pretrained model weights

### 2. **Vector Store & Embeddings** (`embeddings.py`, `vector_store.py`)
- **Embedding Model**: Semantic embeddings for memory and search
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Purpose**: Efficient similarity search for semantic retrieval
- **Indexing Strategy**: Inverted index with vector quantization

**Features**:
- Fast approximate nearest neighbor search
- Scalable to millions of vectors
- Supports incremental updates
- GPU acceleration available

**Files**:
- `embeddings.py` - Embedding generation and management
- `vector_store.py` - FAISS index operations
- `data/memory/faiss.index` - Serialized vector index

### 3. **Search Architecture** (`search_service.py`, `search_engine/search.py`)

#### Components:
- **Web Crawler** (`crawler/worker.py`, `crawler/scheduler.py`)
  - Distributed web crawling with scheduling
  - URL queue management
  - Content extraction and cleaning

- **Hybrid Search** (`indexing/tfidf.py`, `search_engine/search.py`)
  - TF-IDF indexing for lexical search
  - Vector search for semantic queries
  - Ranking combination via PageRank

- **Page Ranking** (`ranking/pagerank.py`)
  - Graph-based importance scoring
  - Link analysis and authority computation

**Search Flow**:
```
User Query
    ↓
[Preprocessing] → Tokenization, normalization
    ↓
┌─────────────────────────────────┐
│   Parallel Search Execution     │
├─────────────┬───────────────────┤
│ Vector      │ TF-IDF Search     │
│ Search      │ (Lexical)         │
└─────────────┴───────────────────┘
    ↓
[Ranking] → PageRank scoring, fusion
    ↓
[Post-Processing] → Citation formatting, deduplication
    ↓
Ranked Results
```

### 4. **Memory Management** (`memory.py`)
- **Chat History**: Conversation context storage
- **User Context**: Name, personality, preferences
- **Retrieval**: Context-aware message fetching
- **Persistence**: Database + vector store backup

### 5. **Data Processing Pipeline**

**Raw Data** → **Preprocessing** → **Tokenization** → **Embedding** → **Index**

Components:
- `preprocess.py` - Text cleaning, normalization
- `dataset.py` - Dataset loading and splitting
- `download_data.py` - Bootstrap data acquisition
- `build_index.py` - Index creation pipeline

### 6. **Database Layer** (`database.py`)
- Stores user profiles, conversation history, preferences
- Chat context retrieval
- Feedback storage
- User session management

### 7. **Learning & Feedback** (`learner.py`, `academic_test.py`, `smoke_test.py`)
- Continuous learning from user interactions
- RLHF (Reinforcement Learning from Human Feedback) integration
- Model evaluation and testing
- Performance monitoring

---

## Frontend Components

### Pages & Purposes

#### 1. **index.html** - Main AI Interface (283 lines)
**Purpose**: Primary entry point - interactive AI chat with mode selection

**Key Features**:
- Unified AI header with logo and action buttons
- Mode selector (5 primary + 4 expandable modes)
- Chat messages area with auto-scrolling
- Resizable textarea input with 120px max-height
- URL query parameter support for pre-filled messages
- Keyboard shortcuts (Ctrl+N for new chat, Ctrl+Enter to send)

**Interaction Flow**:
```
User Selects Mode
    ↓
[Chat/Code/Research/Creative/Analyze]
    ↓
Expands to detailed options
    ↓
Creates appropriate API endpoint
    ↓
Sends request with context
    ↓
Displays streamed/formatted response
```

#### 2. **chat.html** - Dedicated Chat Interface (212 lines)
**Purpose**: Focused conversation mode with unified header

**Features**:
- Streamlined chat layout
- Auto-resizing textarea (min 32px, max 120px)
- Send message via `/chat` endpoint
- More menu with navigation
- Message history display

#### 3. **search.html** - Research & Discovery Interface
**Purpose**: Hybrid search for exploring topics

**Features**:
- Query input with academic mode toggle
- Peer-reviewed filter option
- Results display with ranking indicators
- Citation generation in APA/MLA/Chicago formats
- MathJax support for formula rendering

#### 4. **profile.html** - Personalization (Updated)
**Purpose**: Customize AI personality and preferences

**Section**:
- User name input
- Personality/tone description
- Save preferences locally

**Data Storage**: localStorage + backend sync

#### 5. **about.html** - System Information (Updated)
**Purpose**: Explain core features

**Content**:
- 🧠 Semantic Memory explanation
- 🔍 Hybrid Search description
- 📚 Multi-Mode AI overview
- 🎓 Academic Features summary

#### 6. **feedback.html** - Feedback Collection (Updated)
**Purpose**: Gather user insights

**Form Elements**:
- Star rating (1-5)
- Feedback textarea
- POST to `/feedback` endpoint

#### 7. **history.html** - Conversation History (Updated)
**Purpose**: Browse past interactions

**Display**:
- Date-grouped conversations
- Query preview truncation
- Click to restore conversation
- localStorage backed

### Static Assets

#### **style.css** (2070 lines)
**Architecture**: CSS Variables + Component Classes

**Key Sections**:
- **Root Variables**: Colors (--bg, --ink, --accent), spacing
- **AI Header** (`.ai-header`): Unified navigation
- **Chat Area** (`.ai-chat-area`): Message display
- **Messages** (`.ai-message-wrapper`, `.ai-message-content`): Styling with animations
- **Mode Selector** (`.ai-mode-selector`, `.ai-mode-btn`): Dynamic mode switching
- **Input Section** (`.ai-input-section`, `.ai-textarea`): Resizable compact textarea
- **Dropdowns** (`.ai-dropdown-menu`, `.ai-dropdown-item`): Contextual navigation
- **Responsive Design**: Mobile-first with breakpoints at 768px, 1024px
- **Animations**: Fade-in, slide-in effects for smooth UX

**Color Scheme**:
```css
--bg: Dark background
--panel: Card/container background
--panel-2: Hover state background
--ink: Text color (high contrast)
--muted: Secondary text
--accent: Primary action color
--border: Subtle divider lines
```

#### **app.js** (521 lines)
**Purpose**: Shared JavaScript across all pages

**Core Functions**:

1. **`initAIInterface()`** (~100 lines)
   - Mode switching logic
   - Message input handling
   - API endpoint routing
   - Auto-resize textarea implementation
   - Event listener setup

2. **`addAIMessage(text, role, mode)`**
   - Message DOM creation
   - Role differentiation (user/ai)
   - Auto-scroll to latest message
   - Animation triggering

3. **Preserved Legacy Functions**:
   - `renderHistory(userId)` - Restore conversation
   - `runSearch(query)` - Execute search
   - `initSearchPage()` - Search page setup
   - `getUserId()` - Session management
   - `loadUserProfile()` - Restore preferences

4. **Event Handlers**:
   - Keyboard shortcuts (Enter to send, Ctrl+N for new)
   - Mode button clicks
   - Menu toggles
   - Textarea resizing

---

## Features

### 1. **Multi-Mode AI**
| Mode | Purpose | Endpoint | Use Case |
|------|---------|----------|----------|
| Chat | Conversational AI | `/chat` | General discussion, Q&A |
| Code | Programming help | `/chat` + Code flag | Debugging, implementation |
| Research | Information gathering | `/search` | Academic work, exploration |
| Creative | Writing assistance | `/chat` + Creative flag | Content creation, brainstorming |
| Analyze | Data & text analysis | `/chat` + Analysis flag | Insights, pattern recognition |

### 2. **Semantic Memory**
- Conversation history stored in vector space
- Context retrieval for coherent multi-turn dialogs
- Automatic summarization of long conversations
- Time-aware memory decay (older context weighted less)

### 3. **Hybrid Search**
- **Vector Search**: Semantic understanding of intent
- **TF-IDF Search**: Keyword matching exactness
- **Fusion**: Combined ranking for best results
- **Live Web**: Fallback to current information when needed

### 4. **Academic Features**
- Automatic citation formatting (APA, MLA, Chicago)
- Peer-reviewed content filtering
- Formula rendering with MathJax
- Source attribution with links

### 5. **Personalization**
- User profile with name and personality
- Adaptive response tone based on preference
- Conversation history management
- Custom system prompts

### 6. **Privacy & Local Storage**
- Conversation history in browser localStorage
- No cloud sync without explicit permission
- User data stays local by default
- Optional backend sync for cross-device access

---

## API Endpoints

### Core Routes (Flask Backend)

```
├── GET  /                    # Home page (index.html)
├── GET  /chat-ui             # Chat interface (chat.html)
├── GET  /explore             # Search interface (search.html)
├── GET  /profile-ui          # Profile page (profile.html)
├── GET  /history             # History page (history.html)
├── GET  /about               # About page (about.html)
├── GET  /feedback            # Feedback form (feedback.html)
│
├── POST /chat                # Send chat message
│   ├─ Input: { mode, message, userContext, conversationId }
│   ├─ Process: Route by mode, retrieve context, generate response
│   └─ Output: { response, conversationId, sourceLinks, metadata }
│
├── POST /search              # Perform hybrid search
│   ├─ Input: { query, filters, academicMode, limit }
│   ├─ Process: Vector search + TF-IDF + ranking
│   └─ Output: { results[], ranking[], citations, newSearch }
│
├── GET  /profile             # Retrieve user profile
│   └─ Output: { userId, name, personality, preferences }
│
├── POST /profile             # Update user profile
│   ├─ Input: { name, personality, preferences }
│   └─ Output: { success, updated_fields }
│
├── GET  /history             # Get conversation history
│   ├─ Input: { userId, limit, offset }
│   └─ Output: { conversations[], total_count, nextOffset }
│
├── POST /feedback            # Submit feedback
│   ├─ Input: { response, rating, userId, timestamp }
│   └─ Output: { success, feedbackId }
│
└── GET  /api/*               # Additional API endpoints for AJAX calls
```

### Response Format

**Standard Success Response**:
```json
{
  "success": true,
  "data": { /* response data */ },
  "metadata": {
    "timestamp": "2026-03-22T10:30:00Z",
    "processingTime": 245,
    "model": "gptmini-v1.0"
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "Error description",
  "code": "ERROR_CODE",
  "metadata": { /* metadata */ }
}
```

---

## File Structure

```
LittlefoxAI/
│
├── backend/                           # Python Flask backend
│   ├── app.py                        # Flask app initialization
│   ├── config.py                     # Configuration settings
│   ├── database.py                   # Database operations
│   ├── model.py                      # GPTMini architecture
│   ├── embeddings.py                 # Vector embeddings
│   ├── vector_store.py               # FAISS operations
│   ├── search_service.py             # Search orchestration
│   ├── memory.py                     # Chat memory management
│   ├── learner.py                    # Learning pipeline
│   ├── responder.py                  # Response generation
│   ├── postprocess.py                # Output formatting
│   ├── preprocess.py                 # Input normalization
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── ai/                          # AI & ML modules
│   │   ├── agent.py                 # Agent orchestration
│   │   └── rlhf.py                  # RLHF training
│   │
│   ├── crawler/                     # Web crawling
│   │   ├── scheduler.py             # Task scheduling
│   │   └── worker.py                # Crawling logic
│   │
│   ├── indexing/                    # Search indexing
│   │   ├── embeddings.py            # Embedding indexing
│   │   └── tfidf.py                 # TF-IDF indexing
│   │
│   ├── ranking/                     # Ranking algorithms
│   │   └── pagerank.py              # PageRank implementation
│   │
│   ├── search_engine/               # Search operations
│   │   └── search.py                # Unified search
│   │
│   ├── data/                        # Data storage
│   │   ├── learning_data.txt        # Training data
│   │   ├── personality.txt          # System personality
│   │   ├── seeds.txt                # Crawl seeds
│   │   ├── memory/                  # Vector store
│   │   │   └── faiss.index          # FAISS serialized index
│   │   ├── processed/               # Preprocessed data
│   │   │   ├── cleaned.txt
│   │   │   └── tokenizer.json
│   │   ├── raw/                     # Raw input data
│   │   │   └── data.txt
│   │   └── search_index/            # Search index serialized
│   │
│   ├── saved_model/                 # Model artifacts
│   │   └── model.pth                # Model weights
│   │
│   └── tests/
│       ├── academic_test.py         # Academic feature tests
│       ├── smoke_test.py            # Smoke tests
│       └── test_*.py                # Unit tests
│
├── frontend/                        # React/HTML frontend
│   ├── index.html                   # Main AI interface (283 lines)
│   ├── chat.html                    # Chat interface (212 lines)
│   ├── search.html                  # Search interface
│   ├── profile.html                 # Profile page
│   ├── about.html                   # About page
│   ├── feedback.html                # Feedback form
│   ├── history.html                 # History page
│   │
│   └── static/
│       ├── style.css                # Unified UI styles (2070 lines)
│       └── app.js                   # Shared functionality (521 lines)
│
└── data/                            # Root data folder
    ├── memory/                      # Semantic memory
    ├── processed/                   # Processed artifacts
    │   └── tokenizer.json
    └── search_index/                # Search index storage
```

---

## Data Flow

### 1. **Chat Message Flow**

```
User Input
    ↓
[Frontend] Capture message in textarea
    ↓
[App.js] sendMessage() triggers
    ↓
[POST /chat] Send to backend with context
    ├─ mode: selected interaction mode
    ├─ message: user input text
    ├─ userId: session identifier
    └─ conversationId: current thread
    ↓
[Backend Processing]
├─ Retrieve conversation context from memory
├─ Augment with semantic context from vector store
├─ Preprocess user message (tokenize, normalize)
├─ Route to appropriate handler (Chat/Code/Research/etc)
├─ Generate response with GPTMini
└─ Postprocess output (format URLs, citations, etc)
    ↓
[Response Package]
├─ response_text: Generated message
├─ source_links: Referenced documents
├─ metadata: model, confidence, tokens_used
└─ conversationId: Updated context ID
    ↓
[Frontend] Display message with streaming animation
    ↓
[LocalStorage] Save both user and AI messages
    ↓
[Vector Store] Optionally embed and index for future retrieval
```

### 2. **Search Query Flow**

```
User Query
    ↓
[Search.html] Input submission
    ↓
[POST /search] Send query with filters
├─ query: search string
├─ academicMode: boolean
├─ filters: peer-reviewed, date range, etc
└─ limit: result count
    ↓
[Hybrid Search Engine]
├─ [Vector Search]
│  ├─ Embed query into vector space
│  ├─ Query FAISS index
│  ├─ Retrieve ~100 nearest neighbors
│  └─ Score by relevance
│
└─ [TF-IDF Search]
   ├─ Tokenize query
   ├─ Lookup inverted index
   ├─ Compute BM25 scores
   └─ Retrieve ~100 top docs
    ↓
[Ranking & Fusion]
├─ Combine scores (40% vector, 60% TF-IDF weights)
├─ Apply PageRank boost
├─ Filter by academic constraints if enabled
└─ Deduplicate results
    ↓
[Post-Processing]
├─ Extract summaries
├─ Generate citations (APA/MLA/Chicago)
├─ Format for display
└─ Prepare source links
    ↓
[Return Ranked Results]
└─ Display with score indicators
```

### 3. **Profile Personalization Flow**

```
User Edits Profile
    ↓
[profile.html] Form submission
    ↓
[POST /profile] Update user record
├─ name: user's preferred name
└─ personality: tone/style description
    ↓
[Backend]
├─ Validate input
├─ Update user record in database
├─ Update embedded preferences
└─ Clear relevant caches
    ↓
[Frontend]
├─ Update localStorage cache
├─ Pre-populate future messages with personality
└─ Show confirmation message
```

---

## Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | Flask (Python) | HTTP server, routing |
| Model | GPTMini | Language generation |
| Embeddings | Custom transformer | Semantic vectors |
| Vector DB | FAISS | Similarity search |
| Indexing | TF-IDF + BPE | Full-text + semantic |
| Data | SQLite/JSON | Persistent storage |
| ML | PyTorch | Model training, inference |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Markup | HTML5 | Page structure |
| Styling | CSS3 | UI/UX design |
| Scripting | Vanilla JavaScript (ES6) | Interactivity |
| Icons | FontAwesome 6.4.0 | Visual elements |
| Math | MathJax | Formula rendering |
| Storage | localStorage | Client-side persistence |

### Development
| Tool | Purpose |
|------|---------|
| Git | Version control |
| VS Code | Code editor |
| Python 3.9+ | Runtime environment |
| pip | Package management |
| Node.js (optional) | Build tooling |

---

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Flask and dependencies (see `requirements.txt`)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- 4GB+ RAM for FAISS indexing

### Installation

1. **Clone Repository**
```bash
git clone <repo-url>
cd LittlefoxAI
```

2. **Install Backend Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

3. **Initialize Data**
```bash
python build_index.py        # Build FAISS vector index
python -m spacy download en  # Download language model
```

4. **Start Backend Server**
```bash
python app.py
```
Server runs on `http://localhost:5000`

5. **Access Frontend**
Open browser to `http://localhost:5000`
- Main Interface: `http://localhost:5000/` (index.html)
- Chat Mode: `http://localhost:5000/chat-ui` (chat.html)
- Research: `http://localhost:5000/explore` (search.html)
- Profile: `http://localhost:5000/profile-ui` (profile.html)

### Configuration

**Backend Config** (`backend/config.py`):
```python
DEBUG = True|False                    # Development/production mode
MODEL_PATH = "saved_model/model.pth"  # Model file location
DATA_PATH = "data/"                   # Data directory
VECTOR_DB_PATH = "data/memory/"       # Vector store location
SEARCH_INDEX_PATH = "data/search_index/"
```

**Frontend Config** (`static/app.js`):
- Modify API endpoints if backend hosted elsewhere
- Adjust auto-resize max-height (currently 120px)
- Configure available modes and commands

### Testing

```bash
# Run smoke tests
cd backend
python smoke_test.py

# Run academic feature tests
python academic_test.py

# Test with provided dataset
python dataset.py
```

---

## Performance & Optimization

### Speed Benchmarks
- Chat response: 200-500ms
- Search query: 300-800ms
- Profile load: <50ms
- Page load: <2s (with caching)

### Optimization Tips

1. **Vector Search**: Use GPU acceleration
   ```python
   # In vector_store.py
   index = faiss.index_factory_gpu(dimension, "IVF1K,Flat")
   ```

2. **Cache Management**: Reduce redundant embeddings
   - Cache frequent queries
   - Batch embed operations

3. **Frontend**: Lazy load components
   - Code split by page
   - Defer analytics scripts

4. **Memory**: Implement conversation pruning
   - Keep last 20 messages in context window
   - Archive older conversations to disk

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| FAISS index not found | Run `python build_index.py` |
| Model loading fails | Check `saved_model/model.pth` exists |
| Search returns empty | Verify `search_index/` populated |
| Slow responses | Check CPU usage, consider GPU |
| localStorage full | Clear browser cache (Settings) |

---

## Future Enhancements

- [ ] Multi-language support (Chinese, Spanish, French)
- [ ] Voice input/output integration
- [ ] Image generation via DALL-E API
- [ ] Real-time collaboration features
- [ ] Mobile app (React Native)
- [ ] Knowledge base import from URLs
- [ ] Custom fine-tuning per user
- [ ] Conversation branching/forking
- [ ] Advanced analytics dashboard
- [ ] Plugin architecture for extensions

---

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and test locally
3. Commit with clear messages: `git commit -m "Add feature X"`
4. Push and create Pull Request

---

## License

This project is proprietary. All rights reserved.

---

## Support & Contact

For issues, questions, or suggestions:
- 📧 Email: support@littlefoxai.com
- 🐛 Bug Reports: Issues section
- 💬 Discussions: Discussions board
- 📖 Documentation: `/docs` folder

---

**Last Updated**: March 22, 2026  
**Version**: 1.0.0  
**Status**: Production Ready
