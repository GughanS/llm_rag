import { useState, useRef, useEffect } from 'react';
import './index.css';

// Using the hardcoded development API Key
// In production, this would be managed securely via sessions
const API_KEY = 'test_sk_12345';
const API_URL = 'http://127.0.0.1:8000/generate';

function App() {
  const [model, setModel] = useState('base');
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hello! I am the LLM-RAG model. How can I help you today?', latency: null }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userText = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userText }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/${model}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
        },
        body: JSON.stringify({
          prompt: userText,
          max_new_tokens: 100,
          temperature: 0.7,
        }),
      });

      if (!response.ok) {
        if (response.status === 429) throw new Error("Rate limit exceeded. Please wait a moment.");
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        text: data.response || "No response generated (using random weights).",
        latency: data.latency_ms,
        tokens: data.token_count
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        text: `Error: ${err.message}`, 
        isError: true 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar glass-panel">
        <div className="sidebar-header">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
          </svg>
          LLM-RAG Engine
        </div>
        
        <div className="model-selector">
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Model Selection
          </span>
          <button 
            className={`model-btn ${model === 'base' ? 'active' : ''}`}
            onClick={() => setModel('base')}
          >
            Base Model (GPT-2)
          </button>
          <button 
            className={`model-btn ${model === 'aligned' ? 'active' : ''}`}
            onClick={() => setModel('aligned')}
          >
            DPO Aligned Model
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="main-chat">
        <div className="chat-history">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div style={{ color: msg.isError ? '#ff6b6b' : 'inherit' }}>
                {msg.text}
              </div>
              {msg.latency && (
                <div className="message-meta">
                  {msg.tokens} tokens • {(msg.latency / 1000).toFixed(2)}s
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="message assistant">
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message LLM-RAG..."
              disabled={isLoading}
            />
            <button 
              className="send-btn" 
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
