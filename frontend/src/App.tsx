import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { useEffect, useRef, useState } from "react";

import SpatialBackground from "./components/SpatialBackground";

import "./App.css";

type Message = {
  role: "user" | "assistant";
  content: string;
};

type IconName = "me" | "projects" | "skills" | "fun" | "contact";

/* Small inline icons so the quick-nav looks identical on every platform. */
function NavIcon({ name }: { name: IconName }) {
  const common = {
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  switch (name) {
    case "me":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M8.5 14.2c.9 1.2 2.1 1.8 3.5 1.8s2.6-.6 3.5-1.8" />
          <circle cx="9" cy="10" r="0.6" fill="currentColor" />
          <circle cx="15" cy="10" r="0.6" fill="currentColor" />
        </svg>
      );
    case "projects":
      return (
        <svg {...common}>
          <rect x="3" y="7" width="18" height="13" rx="2.5" />
          <path d="M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7" />
          <path d="M3 12.5h18" />
        </svg>
      );
    case "skills":
      return (
        <svg {...common}>
          <path d="M12 3 3 8l9 5 9-5-9-5Z" />
          <path d="m3 12 9 5 9-5" />
          <path d="m3 16 9 5 9-5" />
        </svg>
      );
    case "fun":
      return (
        <svg {...common}>
          <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" />
          <path d="M19 16l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7.7-2Z" />
        </svg>
      );
    case "contact":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="4" />
          <path d="M16 8v5a3 3 0 0 0 6 0v-1a10 10 0 1 0-3.92 7.94" />
        </svg>
      );
  }
}

function App() {
  const [question, setQuestion] = useState("");

  const [messages, setMessages] = useState<Message[]>([]);

  const [loading, setLoading] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);

  const sessionId = useRef(crypto.randomUUID());

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  async function askAI(customQuestion?: string) {
    const currentQuestion = customQuestion || question;

    if (!currentQuestion.trim() || loading) {
      return;
    }

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: currentQuestion,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch("https://my-portfolio-chatbot.onrender.com/chat", {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          question: currentQuestion,

          session_id: sessionId.current,
        }),
      });

      if (!response.ok) {
        throw new Error("Backend error");
      }

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();

      const decoder = new TextDecoder();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: "",
        },
      ]);

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        const chunk = decoder.decode(value, {
          stream: true,
        });

        setMessages((previous) => {
          const updated = [...previous];

          const lastIndex = updated.length - 1;

          updated[lastIndex] = {
            ...updated[lastIndex],

            content: updated[lastIndex].content + chunk,
          };

          return updated;
        });
      }
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",

          content: "Could not connect to the backend.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function clearChat() {
    try {
      await fetch(`https://my-portfolio-chatbot.onrender.com/chat/${sessionId.current}`, {
        method: "DELETE",
      });

      setMessages([]);
      setQuestion("");
    } catch (error) {
      console.error("Could not clear chat");
    }
  }

  function askPreset(questionText: string) {
    askAI(questionText);
  }

  const inConversation = messages.length > 0 || loading;

  /* Show the "Thinking" card only until the first streamed text arrives. */
  const lastMessage = messages[messages.length - 1];
  const showThinking =
    loading &&
    !(
      lastMessage &&
      lastMessage.role === "assistant" &&
      lastMessage.content.length > 0
    );

  return (
    <div className="app">
      <SpatialBackground />

      {/* Huge faint name, clipped by the bottom of the viewport */}
      {!inConversation && (
        <div className="ghost-name" aria-hidden="true">
          Mayank
        </div>
      )}

      <div className="page-shell">
        <header className="topbar">
          <button
            className="availability"
            onClick={() =>
              askPreset("Why should a recruiter consider Mayank?")
            }
          >
            <span />
            Looking for a talent?
          </button>

          <a
            className="github-link"
            href="https://github.com/mynk7708"
            target="_blank"
            rel="noreferrer"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path
                fill="currentColor"
                d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.28-5.27-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.47.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.58.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.71 5.39-5.29 5.68.42.36.79 1.07.79 2.16v3.2c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z"
              />
            </svg>

            <span className="github-label">GitHub Repo</span>
          </a>
        </header>

        {!inConversation && (
          <main className="landing">
            <div className="monogram">
              <span>M</span>
            </div>

            <p className="intro-line">
              Hey, I’m Mayank
              <span className="wave" aria-hidden="true">
                👋
              </span>
            </p>

            <h1>AI Portfolio</h1>

            <p className="landing-subtitle">
              A conversational window into my work, skills and journey.
            </p>

            <div className="hero-composer">
              <input
                type="text"
                value={question}
                placeholder="Ask me anything..."
                aria-label="Ask me anything"
                autoComplete="off"
                enterKeyHint="send"
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    askAI();
                  }
                }}
              />

              <button
                onClick={() => askAI()}
                disabled={loading}
                aria-label="Send question"
              >
                →
              </button>
            </div>

            <nav className="quick-nav">
              <button onClick={() => askPreset("Tell me about Mayank.")}>
                <span className="nav-icon nav-me">
                  <NavIcon name="me" />
                </span>
                <strong>Me</strong>
                <small>Who I am</small>
              </button>

              <button
                onClick={() => askPreset("Tell me about Mayank's projects.")}
              >
                <span className="nav-icon nav-projects">
                  <NavIcon name="projects" />
                </span>
                <strong>Projects</strong>
                <small>What I’ve built</small>
              </button>

              <button onClick={() => askPreset("What are Mayank's skills?")}>
                <span className="nav-icon nav-skills">
                  <NavIcon name="skills" />
                </span>
                <strong>Skills</strong>
                <small>My toolkit</small>
              </button>

              <button
                onClick={() =>
                  askPreset(
                    "Tell me something interesting about Mayank's interests and personality."
                  )
                }
              >
                <span className="nav-icon nav-fun">
                  <NavIcon name="fun" />
                </span>
                <strong>Fun</strong>
                <small>Beyond code</small>
              </button>

              <button
                onClick={() => askPreset("How can someone contact Mayank?")}
              >
                <span className="nav-icon nav-contact">
                  <NavIcon name="contact" />
                </span>
                <strong>Contact</strong>
                <small>Let’s connect</small>
              </button>
            </nav>
          </main>
        )}

        {inConversation && (
          <main className="conversation">
            <div className="conversation-top">
              <div>
                <span className="conversation-kicker">Mayank AI</span>

                <h2>Conversation</h2>
              </div>

              <button className="clear-chat" onClick={clearChat}>
                New conversation
              </button>
            </div>

            <section className="chat-stream">
              {messages.map((message, index) => {
                /* Hide the empty assistant placeholder while "Thinking" shows. */
                if (message.role === "assistant" && message.content === "") {
                  return null;
                }

                /* Number questions 01, 02, 03... (not every message). */
                const questionNumber = messages
                  .slice(0, index + 1)
                  .filter((m) => m.role === "user").length;

                return (
                  <article
                    className={`message ${message.role}`}
                    key={index}
                  >
                    {message.role === "user" ? (
                      <div className="question-card">
                        <span className="question-index">
                          {String(questionNumber).padStart(2, "0")}
                        </span>

                        <span className="question-label">You asked</span>

                        <blockquote>{message.content}</blockquote>
                      </div>
                    ) : (
                      <div className="answer-card">
                       <div className="answer-heading">
  <div className="mini-mark" aria-hidden="true">
    <svg viewBox="0 0 48 48">
      <path
        className="mark-frame"
        d="M24 3.5 42.5 14v20L24 44.5 5.5 34V14L24 3.5Z"
      />

      <path
        className="mark-letter"
        d="M13.5 31.5 18 16.8l6 9.4 6-9.4 4.5 14.7"
      />

      <circle
        className="mark-dot"
        cx="38.2"
        cy="9.8"
        r="2.2"
      />
    </svg>
  </div>

  <div className="answer-identity">
    <div className="answer-name-row">
      <strong>Mayank AI</strong>
      <span
        className="verified-dot"
        aria-hidden="true"
      />
    </div>

  </div>
</div>

                        <div className="message-content">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </article>
                );
              })}

              {showThinking && (
                <div className="thinking-card">
                  <div className="thinking-orb">
                    <span />
                    <span />
                    <span />
                  </div>

                  <div>
                    <strong>Thinking</strong>

                    
                      Thinking...
                    
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </section>
          </main>
        )}

        {inConversation && (
          <div className="floating-composer">
            <input
              type="text"
              value={question}
              placeholder="Ask a follow-up..."
              aria-label="Ask a follow-up"
              autoComplete="off"
              enterKeyHint="send"
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  askAI();
                }
              }}
            />

            <button
              onClick={() => askAI()}
              disabled={loading}
              aria-label="Send follow-up"
            >
              →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;