import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Loader2, Send, Sparkles } from 'lucide-react';
import { trpc } from '@/lib/trpc';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    intent?: string;
    confidence?: number;
    modelUsed?: string;
  };
}

export default function GeminiAgent() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m Helix! I can help you with COVID-19 predictions or customer churn analysis. What would you like to know?',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const agentQueryMutation = trpc.agent.query.useMutation();

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
      // Call the agent API using tRPC
      const data = await agentQueryMutation.mutateAsync({
        query: currentInput,
      });

      // Add assistant response
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        metadata: {
          intent: data.intent,
          confidence: data.confidence,
          modelUsed: data.modelUsed,
        },
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const exampleQueries = [
    'What\'s the COVID-19 situation in the United States?',
    'Is customer CUST_001 likely to churn?',
    'Analyze COVID-19 risk for France with 500,000 cases',
    'Check churn probability for a customer with 24 months tenure',
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Sparkles className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-slate-900">HELIX</h1>
          </div>
          <p className="text-slate-600">
            Ask me about COVID-19 predictions or customer churn analysis
          </p>
        </div>

        {/* Main Chat Container */}
        <Card className="flex flex-col h-[600px] shadow-lg">
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 1 && (
              <div className="space-y-3">
                <p className="text-sm text-slate-600 font-medium">Try asking about:</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {exampleQueries.map((query, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setInput(query);
                        inputRef.current?.focus();
                      }}
                      className="text-left p-3 rounded-lg border border-slate-200 hover:border-blue-400 hover:bg-blue-50 transition-colors text-sm text-slate-700 hover:text-blue-700"
                    >
                      {query}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl px-4 py-3 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white rounded-br-none'
                      : 'bg-slate-200 text-slate-900 rounded-bl-none'
                  }`}
                >
                  <p className="text-sm md:text-base whitespace-pre-wrap">{message.content}</p>
                  {message.metadata && (
                    <div className="mt-2 pt-2 border-t border-opacity-20 border-current text-xs opacity-75">
                      {message.metadata.intent && (
                        <p>Intent: <span className="font-semibold">{message.metadata.intent}</span></p>
                      )}
                      {message.metadata.confidence && (
                        <p>Confidence: <span className="font-semibold">{(message.metadata.confidence * 100).toFixed(1)}%</span></p>
                      )}
                      {message.metadata.modelUsed && (
                        <p>Model: <span className="font-semibold">{message.metadata.modelUsed}</span></p>
                      )}
                    </div>
                  )}
                  <span className="text-xs opacity-70 mt-1 block">
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-slate-200 text-slate-900 px-4 py-3 rounded-lg rounded-bl-none">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">Thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-slate-200 p-4 bg-white rounded-b-lg">
            <form onSubmit={handleSendMessage} className="flex gap-2">
              <Input
                ref={inputRef}
                type="text"
                placeholder="Ask about COVID-19 or customer churn..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading}
                className="flex-1"
              />
              <Button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </form>
          </div>
        </Card>

        {/* Info Section */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="p-4">
            <h3 className="font-semibold text-slate-900 mb-2">COVID-19 Analysis</h3>
            <p className="text-sm text-slate-600">
              Get risk assessments and predictions based on epidemiological data
            </p>
          </Card>
          <Card className="p-4">
            <h3 className="font-semibold text-slate-900 mb-2">Churn Prediction</h3>
            <p className="text-sm text-slate-600">
              Analyze customer retention risk and get retention recommendations
            </p>
          </Card>
          <Card className="p-4">
            <h3 className="font-semibold text-slate-900 mb-2">AI-Powered</h3>
            <p className="text-sm text-slate-600">
              Powered by Gemini AI for intelligent intent classification and routing
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}
