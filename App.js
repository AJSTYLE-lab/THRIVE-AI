import React from "react";
import ChatBox from "./components/ChatBox";
import "./App.css";

const App = () => {
  return (
    <div className="app-container">
      <header>
        <h1>🧠 THRIVE-AI</h1>
        <p>
          Your empathetic RAG-powered companion for mental well-being. All chats are private and safe.
        </p>
      </header>
      <ChatBox />
      <footer>
        <p>© 2025 THRIVE-AI | Built with ❤️ for mental wellness.</p>
      </footer>
    </div>
  );
};

export default App;
