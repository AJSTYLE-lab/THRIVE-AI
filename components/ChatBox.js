// src/components/ChatBox.js
import React, { useState } from "react";
import { FaGlobe } from "react-icons/fa";
import "./ChatBox.css";

const GEMINI_API_KEY = "AIzaSyBJLDqODddRy8QfTyQyxGGxUaWv9kCHFN8";

const formatText = (text) => {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
};

const sendToGemini = async (message) => {
  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contents: [{ parts: [{ text: message }] }] }),
      }
    );
    const data = await response.json();
    return (
      data?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "⚠️ I didn't get that. Can you rephrase?"
    );
  } catch (error) {
    console.error("Gemini API error:", error);
    return "⚠️ Error talking to the assistant.";
  }
};

const ChatBox = ({ setShowDetectionPage, generateReport, setShowMap }) => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");

  const sendMessage = async () => {
    const trimmed = inputText.trim();
    if (!trimmed) return;

    setMessages((prev) => [...prev, { text: trimmed, sender: "user" }]);
    setMessages((prev) => [...prev, { text: "🤖 Thinking...", sender: "bot" }]);
    setInputText("");

    const response = await sendToGemini(trimmed);

    setMessages((prev) => [
      ...prev.slice(0, -1),
      { text: formatText(response), sender: "bot" },
    ]);
  };

  return (
    <div className="chatbox-container">
      <div className="chatbox-messages">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.sender}`}
            dangerouslySetInnerHTML={{ __html: msg.text }}
          />
        ))}
      </div>

      <div className="chatbox-input">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask about mental health..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>➤</button>
      </div>
    </div>
  );
};

export default ChatBox;
