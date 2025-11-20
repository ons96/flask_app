import { VercelRequest, VercelResponse } from '@vercel/node';

const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || '';

interface CharacterCard {
  name?: string;
  description?: string;
  personality?: string;
  scenario?: string;
  system_prompt?: string;
  example_dialogue?: string;
}

interface RoleplayRequest {
  message: string;
  character: CharacterCard;
  model?: string;
  provider?: 'puter' | 'groq' | 'openrouter';
  history?: Array<{ role: string; content: string }>;
}

// Build system prompt from character card
function buildSystemPrompt(character: CharacterCard): string {
  const parts: string[] = [];

  if (character.system_prompt) {
    parts.push(character.system_prompt);
  } else {
    if (character.name) {
      parts.push(`You are ${character.name}.`);
    }
    if (character.description) {
      parts.push(`Description: ${character.description}`);
    }
    if (character.personality) {
      parts.push(`Personality: ${character.personality}`);
    }
    if (character.scenario) {
      parts.push(`Scenario: ${character.scenario}`);
    }
    if (character.example_dialogue) {
      parts.push(`Example of how you speak: ${character.example_dialogue}`);
    }
  }

  return parts.join('\n\n') || 'You are a helpful assistant.';
}

async function roleplayWithPuter(
  message: string,
  systemPrompt: string,
  history: Array<{ role: string; content: string }> = [],
  model: string = 'gpt-4o-mini'
): Promise<string> {
  try {
    const messages = [
      { role: 'system', content: systemPrompt },
      ...history,
      { role: 'user', content: message },
    ];

    const response = await fetch('https://api.puter.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        stream: false,
        temperature: 0.8,
      }),
    });

    if (!response.ok) {
      throw new Error(`Puter API error: ${response.status}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || 'No response';
  } catch (error) {
    throw error;
  }
}

async function roleplayWithGroq(
  message: string,
  systemPrompt: string,
  history: Array<{ role: string; content: string }> = [],
  model: string = 'mixtral-8x7b-32768'
): Promise<string> {
  if (!GROQ_API_KEY) {
    throw new Error('Groq API key not configured');
  }

  try {
    const messages = [
      { role: 'system', content: systemPrompt },
      ...history,
      { role: 'user', content: message },
    ];

    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.8,
      }),
    });

    if (!response.ok) {
      throw new Error(`Groq API error: ${response.status}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || 'No response';
  } catch (error) {
    throw error;
  }
}

async function roleplayWithOpenRouter(
  message: string,
  systemPrompt: string,
  history: Array<{ role: string; content: string }> = [],
  model: string = 'gpt-3.5-turbo'
): Promise<string> {
  if (!OPENROUTER_API_KEY) {
    throw new Error('OpenRouter API key not configured');
  }

  try {
    const messages = [
      { role: 'system', content: systemPrompt },
      ...history,
      { role: 'user', content: message },
    ];

    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://puter-free-chatbot.vercel.app',
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.8,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenRouter API error: ${response.status}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || 'No response';
  } catch (error) {
    throw error;
  }
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
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

  const {
    message,
    character = {},
    model = 'gpt-4o-mini',
    provider = 'puter',
    history = [],
  }: RoleplayRequest = req.body;

  if (!message) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    return res.status(400).json({ error: 'Message is required' });
  }

  try {
    const systemPrompt = buildSystemPrompt(character);
    let response: string;

    switch (provider) {
      case 'groq':
        response = await roleplayWithGroq(message, systemPrompt, history, model);
        break;
      case 'openrouter':
        response = await roleplayWithOpenRouter(message, systemPrompt, history, model);
        break;
      case 'puter':
      default:
        response = await roleplayWithPuter(message, systemPrompt, history, model);
        break;
    }

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');

    res.status(200).json({
      response,
      character_name: character.name || 'Unknown',
      model,
      provider,
    });
  } catch (error) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.status(500).json({
      error: 'Roleplay request failed',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
