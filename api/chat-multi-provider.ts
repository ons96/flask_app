import { VercelRequest, VercelResponse } from '@vercel/node';

// Multi-provider LLM chat endpoint that works on BlackBerry Classic
// Supports: Puter, Groq, OpenRouter (and others via OpenRouter)

const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || '';

interface ChatRequest {
  message: string;
  model?: string;
  provider?: 'puter' | 'groq' | 'openrouter';
  stream?: boolean;
}

interface ChatResponse {
  response: string;
  model: string;
  provider: string;
  tokens?: number;
}

// Provider-specific handlers
async function chatWithPuter(message: string, model: string = 'gpt-4o-mini'): Promise<string> {
  try {
    const response = await fetch('https://api.puter.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: message }],
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Puter API error: ${response.status}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || 'No response';
  } catch (error) {
    throw new Error(`Puter error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function chatWithGroq(message: string, model: string = 'mixtral-8x7b-32768'): Promise<string> {
  if (!GROQ_API_KEY) {
    throw new Error('Groq API key not configured');
  }

  try {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: message }],
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Groq API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || 'No response';
  } catch (error) {
    throw new Error(`Groq error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function chatWithOpenRouter(message: string, model: string = 'gpt-3.5-turbo'): Promise<string> {
  if (!OPENROUTER_API_KEY) {
    throw new Error('OpenRouter API key not configured');
  }

  try {
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://puter-free-chatbot.vercel.app',
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: message }],
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenRouter API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || 'No response';
  } catch (error) {
    throw new Error(`OpenRouter error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message, model = 'gpt-4o-mini', provider = 'puter' }: ChatRequest = req.body;

  if (!message) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    return res.status(400).json({ error: 'Message is required' });
  }

  try {
    let response: string;

    switch (provider) {
      case 'groq':
        response = await chatWithGroq(message, model);
        break;
      case 'openrouter':
        response = await chatWithOpenRouter(message, model);
        break;
      case 'puter':
      default:
        response = await chatWithPuter(message, model);
        break;
    }

    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');

    res.status(200).json({
      response,
      model,
      provider,
    } as ChatResponse);
  } catch (error) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.status(500).json({
      error: 'Chat request failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      provider,
    });
  }
}
