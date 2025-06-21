import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [context, setContext] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [showContextInput, setShowContextInput] = useState(false);

  // Fetch available models on component mount
  useEffect(() => {
    async function fetchModels() {
      try {
        const response = await fetch('http://localhost:8000/models');
        const data = await response.json();
        setAvailableModels(data.models);
        if (data.models.length > 0) {
          setSelectedModel(data.models[0]);
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    }
    fetchModels();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || !selectedModel) return;
    
    setIsLoading(true);
    const userMessage = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_input: input,
          model_name: selectedModel,
          context: context.trim() || undefined
        })
      });
      
      const data = await response.json();
      setMessages(prev => [...prev, { 
        text: data.response, 
        sender: 'bot',
        model: data.model_used
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        text: "Sorry, something went wrong.", 
        sender: 'bot',
        model: 'error'
      }]);
    }
    
    setInput('');
    setIsLoading(false);
  };

  const handleStartNewChat = () => {
    setMessages([]);
    setContext('');
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    handleStartNewChat();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
    // Shift+Enter will work as default (new line)
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Customer Support Chat POC</h1>
        <div className="model-selector">
          <label htmlFor="model-select">Model: </label>
          <select 
            id="model-select"
            value={selectedModel}
            onChange={handleModelChange}
            disabled={isLoading}
          >
            {availableModels.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
          <button 
            className="context-toggle"
            onClick={() => setShowContextInput(!showContextInput)}
          >
            {showContextInput ? 'Hide Context' : 'Add Context'}
          </button>
          <button 
            className="new-chat"
            onClick={handleStartNewChat}
            disabled={isLoading}
          >
            New Chat
          </button>
        </div>
      </header>

      {showContextInput && (
        <div className="context-input">
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="Enter context for the bot (e.g., 'You are a customer support agent for a tech company. The customer is having issues with their router. All personal data should be fictional.')"
            rows={3}
          />
        </div>
      )}

      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.sender}`}>
              <div className="message-content">{msg.text}</div>
              {msg.model && msg.sender === 'bot' && (
                <div className="message-meta">Model: {msg.model}</div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-content">Thinking...</div>
              <div className="message-meta">Model: {selectedModel}</div>
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="chat-input">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (Shift+Enter for new line)"
            disabled={isLoading}
            rows={1}
            style={{ resize: 'none' }}
            onInput={(e) => {
              e.target.style.height = 'auto';
              e.target.style.height = e.target.scrollHeight + 'px';
            }}
          />
          <button type="submit" disabled={isLoading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;