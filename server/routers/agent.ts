import { z } from 'zod';
import { publicProcedure, router } from '../_core/trpc';
import { invokeLLM } from '../_core/llm';
import axios from 'axios';
import type { TextContent, ImageContent, FileContent } from '../_core/llm';

/**
 * Gemini Agent Router
 * Handles intent classification, parameter extraction, and model routing
 */

const AgentQuerySchema = z.object({
  query: z.string().min(1, 'Query cannot be empty'),
});

type AgentQueryInput = z.infer<typeof AgentQuerySchema>;

interface AgentResponse {
  response: string;
  intent: 'covid' | 'churn' | 'unknown';
  confidence: number;
  modelUsed?: string;
  rawPrediction?: Record<string, unknown>;
}

/**
 * Classify user intent using keyword matching
 */
function classifyIntent(query: string): { intent: 'covid' | 'churn' | 'unknown'; confidence: number } {
  const covidKeywords = [
    'covid', 'coronavirus', 'pandemic', 'virus', 'infection',
    'cases', 'deaths', 'vaccination', 'testing', 'outbreak',
    'epidemic', 'disease', 'health', 'country', 'spread'
  ];

  const churnKeywords = [
    'churn', 'customer', 'leave', 'cancel', 'subscription',
    'billing', 'service', 'complaint', 'support', 'contract',
    'retention', 'loyalty', 'telecom', 'internet', 'phone'
  ];

  const queryLower = query.toLowerCase();

  const covidMatches = covidKeywords.filter(kw => queryLower.includes(kw)).length;
  const churnMatches = churnKeywords.filter(kw => queryLower.includes(kw)).length;

  if (covidMatches > churnMatches && covidMatches > 0) {
    return {
      intent: 'covid',
      confidence: Math.min(covidMatches / (covidMatches + churnMatches + 1), 1.0),
    };
  } else if (churnMatches > 0) {
    return {
      intent: 'churn',
      confidence: Math.min(churnMatches / (covidMatches + churnMatches + 1), 1.0),
    };
  }

  return { intent: 'unknown', confidence: 0 };
}

/**
 * Helper function to clean JSON from markdown code blocks
 */
function cleanJsonResponse(content: string): string {
  // Remove markdown code blocks if present
  let cleaned = content.trim();
  if (cleaned.startsWith('```json')) {
    cleaned = cleaned.replace(/^```json\s*/, '').replace(/\s*```$/, '');
  } else if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```\s*/, '').replace(/\s*```$/, '');
  }
  return cleaned.trim();
}

/**
 * Extract COVID-19 parameters from query using LLM
 */
async function extractCovidParameters(query: string): Promise<Record<string, unknown>> {
  try {
    const response = await invokeLLM({
      messages: [
        {
          role: 'system',
          content: 'You are a data extraction assistant. Extract COVID-19 prediction parameters from user queries. Return only valid JSON. ALWAYS extract country names even if mentioned informally (e.g., "United States", "USA", "America" → "USA").',
        },
        {
          role: 'user',
          content: `Extract COVID-19 parameters from this query: "${query}"
          
          Return a JSON object with these fields (use null for missing values):
          {
            "country_name": "country name (REQUIRED - extract from query even if informal)",
            "confirmed_cases": "number or null",
            "deaths": "number or null",
            "recovered": "number or null",
            "population": "number or null",
            "vaccination_rate": "decimal 0-1 or null",
            "testing_rate": "number or null"
          }
          
          Examples:
          - "COVID in USA" → {"country_name": "USA"}
          - "What's the situation in India?" → {"country_name": "India"}
          - "United States COVID" → {"country_name": "USA"}
          
          Only return the JSON object, no other text.`,
        },
      ],
    });

    const messageContent = response.choices?.[0]?.message?.content;
    const content = typeof messageContent === 'string' ? messageContent : '{}';
    const cleanedContent = cleanJsonResponse(content);
    const params = JSON.parse(cleanedContent);
    
    // Fallback: Try to extract country from query using simple pattern matching
    if (!params.country_name || params.country_name === 'null') {
      const countryMatch = query.match(/\b(USA|US|United States|America|India|China|Brazil|UK|United Kingdom|Canada|Australia|Germany|France|Italy|Spain|Japan|Russia)\b/i);
      if (countryMatch) {
        params.country_name = countryMatch[1];
      }
    }
    
    return params;
  } catch (error) {
    console.error('Error extracting COVID parameters:', error);
    // Try simple pattern matching as fallback
    const countryMatch = query.match(/\b(USA|US|United States|America|India|China|Brazil|UK|United Kingdom|Canada|Australia|Germany|France|Italy|Spain|Japan|Russia)\b/i);
    return countryMatch ? { country_name: countryMatch[1] } : {};
  }
}

/**
 * Extract Churn parameters from query using LLM
 */
async function extractChurnParameters(query: string): Promise<Record<string, unknown>> {
  try {
    const response = await invokeLLM({
      messages: [
        {
          role: 'system',
          content: 'You are a data extraction assistant. Extract customer churn prediction parameters from user queries. Return only valid JSON. Extract numbers even if mentioned informally.',
        },
        {
          role: 'user',
          content: `Extract Telco customer churn parameters from this query: "${query}"
          
          Return a JSON object with these fields (use null for missing values):
          {
            "customer_id": "customer ID or null",
            "age": "age or null",
            "tenure_months": "months or null",
            "monthly_charges": "amount or null",
            "total_charges": "amount or null",
            "contract_type": "contract type or null",
            "internet_service_type": "service type or null",
            "tech_support": "true/false or null",
            "online_security": "true/false or null",
            "support_tickets_count": "number or null"
          }
          
          Examples:
          - "customer with 24 months tenure" → {"tenure_months": 24}
          - "paying $85 monthly" → {"monthly_charges": 85}
          - "12 month customer" → {"tenure_months": 12}
          
          Only return the JSON object, no other text.`,
        },
      ],
    });

    const messageContent = response.choices?.[0]?.message?.content;
    const content = typeof messageContent === 'string' ? messageContent : '{}';
    const cleanedContent = cleanJsonResponse(content);
    const params = JSON.parse(cleanedContent);
    
    // Fallback: Try to extract numbers from query using pattern matching
    if (!params.tenure_months) {
      const tenureMatch = query.match(/(\d+)\s*(month|months|mo)/i);
      if (tenureMatch) {
        params.tenure_months = parseInt(tenureMatch[1]);
      }
    }
    
    if (!params.monthly_charges) {
      const chargesMatch = query.match(/\$?(\d+(?:\.\d+)?)\s*(monthly|per month|\/month)?/i);
      if (chargesMatch) {
        params.monthly_charges = parseFloat(chargesMatch[1]);
      }
    }
    
    return params;
  } catch (error) {
    console.error('Error extracting Churn parameters:', error);
    // Try simple pattern matching as fallback
    const params: Record<string, unknown> = {};
    const tenureMatch = query.match(/(\d+)\s*(month|months|mo)/i);
    if (tenureMatch) {
      params.tenure_months = parseInt(tenureMatch[1]);
    }
    const chargesMatch = query.match(/\$?(\d+(?:\.\d+)?)/);
    if (chargesMatch) {
      params.monthly_charges = parseFloat(chargesMatch[1]);
    }
    return params;
  }
}

/**
 * Call COVID-19 prediction service
 */
async function callCovidService(parameters: Record<string, unknown>): Promise<Record<string, unknown>> {
  try {
    const covidServiceUrl = process.env.COVID_SERVICE_URL || 'http://localhost:8000';
    const response = await axios.post(`${covidServiceUrl}/predict/covid`, parameters, {
      timeout: 30000,
    });
    return response.data;
  } catch (error) {
    console.error('Error calling COVID service:', error);
    return { error: 'Failed to get COVID prediction' };
  }
}

/**
 * Call Churn prediction service
 */
async function callChurnService(parameters: Record<string, unknown>): Promise<Record<string, unknown>> {
  try {
    const churnServiceUrl = process.env.CHURN_SERVICE_URL || 'http://localhost:8001';
    const response = await axios.post(`${churnServiceUrl}/predict/churn`, parameters, {
      timeout: 30000,
    });
    return response.data;
  } catch (error) {
    console.error('Error calling Churn service:', error);
    return { error: 'Failed to get churn prediction' };
  }
}

/**
 * Synthesize natural language response from prediction
 */
async function synthesizeResponse(
  intent: 'covid' | 'churn' | 'unknown',
  prediction: Record<string, unknown>
): Promise<string> {
  if ('error' in prediction) {
    return `I encountered an error: ${prediction.error}`;
  }

  try {
    const prompt = intent === 'covid'
      ? `Synthesize a natural language response for this COVID-19 prediction: ${JSON.stringify(prediction)}`
      : `Synthesize a natural language response for this churn prediction: ${JSON.stringify(prediction)}`;

    const response = await invokeLLM({
      messages: [
        {
          role: 'system',
          content: 'You are a helpful assistant that explains ML model predictions in clear, actionable language.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
    });

    const messageContent = response.choices?.[0]?.message?.content;
    return typeof messageContent === 'string' ? messageContent : 'Unable to synthesize response';
  } catch (error) {
    console.error('Error synthesizing response:', error);
    return 'I encountered an error while processing your request.';
  }
}

export const agentRouter = router({
  /**
   * Main agent query endpoint
   * Processes user query through intent classification, parameter extraction, and model routing
   */
  query: publicProcedure
    .input(AgentQuerySchema)
    .mutation(async ({ input }): Promise<AgentResponse> => {
      const { query } = input;

      // Step 1: Classify intent
      const { intent, confidence } = classifyIntent(query);

      if (intent === 'unknown' || confidence < 0.3) {
        return {
          response: "I'm not sure if you're asking about COVID-19 or customer churn. Could you please clarify?",
          intent: 'unknown',
          confidence: 0,
        };
      }

      // Step 2: Extract parameters
      let parameters: Record<string, unknown>;
      if (intent === 'covid') {
        parameters = await extractCovidParameters(query);
        // COVID service can work with minimal information (uses defaults)
        if (!parameters || Object.keys(parameters).length === 0) {
          return {
            response: 'I need at least some information to make a COVID-19 prediction. Please provide country name or case statistics.',
            intent: 'covid',
            confidence,
          };
        }
        // If no country_name provided, use a default
        if (!('country_name' in parameters)) {
          parameters.country_name = 'Unknown';
        }
      } else {
        parameters = await extractChurnParameters(query);
        // Churn service can work with minimal information (generates defaults)
        if (!parameters || Object.keys(parameters).length === 0) {
          return {
            response: 'I need at least some information about the customer to assess churn risk. Please provide details like tenure, monthly charges, or contract type.',
            intent: 'churn',
            confidence,
          };
        }
        // If no customer_id provided, service will generate one
        if (!('customer_id' in parameters)) {
          parameters.customer_id = `GUEST_${Date.now()}`;
        }
      }

      // Step 3: Call appropriate service
      let prediction: Record<string, unknown>;
      if (intent === 'covid') {
        prediction = await callCovidService(parameters);
      } else {
        prediction = await callChurnService(parameters);
      }

      // Step 4: Synthesize response
      const response = await synthesizeResponse(intent, prediction);

      return {
        response,
        intent,
        confidence,
        modelUsed: intent === 'covid' ? 'COVID-19 Prediction Model' : 'Churn Prediction Model',
        rawPrediction: prediction,
      };
    }),

  /**
   * Health check endpoint
   */
  health: publicProcedure.query(() => ({
    status: 'healthy',
    timestamp: new Date().toISOString(),
  })),
});
