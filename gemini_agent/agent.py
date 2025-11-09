"""
Gemini AI Agent for MLOps Dual Model Inference

This module implements an intelligent agent that:
1. Classifies user intent (COVID-19 or Telco Churn)
2. Extracts relevant parameters from natural language queries
3. Routes requests to the appropriate ML model service
4. Synthesizes natural language responses from model predictions

The agent uses Google's Gemini API with function calling capabilities.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiAgent:
    """Intelligent routing agent using Gemini API."""
    
    def __init__(self, api_key: str, covid_service_url: str, churn_service_url: str):
        """
        Initialize the Gemini Agent.
        
        Args:
            api_key: Google Gemini API key
            covid_service_url: URL of COVID-19 prediction service
            churn_service_url: URL of Telco Churn prediction service
        """
        self.api_key = api_key
        self.covid_service_url = covid_service_url
        self.churn_service_url = churn_service_url
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """
        Classify the user's intent as COVID-19 or Telco Churn prediction.
        
        Args:
            query: User's natural language query
            
        Returns:
            Tuple of (intent, confidence) where intent is 'covid' or 'churn'
        """
        logger.info(f"Classifying intent for query: {query}")
        
        # Keywords for intent classification
        covid_keywords = [
            'covid', 'coronavirus', 'pandemic', 'virus', 'infection',
            'cases', 'deaths', 'vaccination', 'testing', 'outbreak',
            'epidemic', 'disease', 'health', 'country', 'spread'
        ]
        
        churn_keywords = [
            'churn', 'customer', 'leave', 'cancel', 'subscription',
            'billing', 'service', 'complaint', 'support', 'contract',
            'retention', 'loyalty', 'telecom', 'internet', 'phone'
        ]
        
        query_lower = query.lower()
        
        # Count keyword matches
        covid_matches = sum(1 for keyword in covid_keywords if keyword in query_lower)
        churn_matches = sum(1 for keyword in churn_keywords if keyword in query_lower)
        
        # Determine intent
        if covid_matches > churn_matches:
            intent = 'covid'
            confidence = min(covid_matches / (covid_matches + churn_matches + 1), 1.0)
        else:
            intent = 'churn'
            confidence = min(churn_matches / (covid_matches + churn_matches + 1), 1.0)
        
        logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")
        return intent, confidence
    
    def extract_covid_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract COVID-19 prediction parameters from user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary of extracted parameters
        """
        logger.info("Extracting COVID-19 parameters...")
        
        # Use Gemini to extract structured parameters
        extraction_prompt = f"""
        Extract COVID-19 prediction parameters from this query: "{query}"
        
        Return a JSON object with these fields (use null for missing values):
        {{
            "country_name": "country name",
            "confirmed_cases": "number of confirmed cases",
            "deaths": "number of deaths",
            "recovered": "number of recovered cases",
            "population": "country population",
            "vaccination_rate": "vaccination rate as decimal (0-1)",
            "testing_rate": "testing rate per 1M population"
        }}
        
        Only return the JSON object, no other text.
        """
        
        params = self._call_gemini(extraction_prompt)
        logger.info(f"Extracted COVID parameters: {params}")
        return params
    
    def extract_churn_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract Telco Churn prediction parameters from user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary of extracted parameters
        """
        logger.info("Extracting Telco Churn parameters...")
        
        # Use Gemini to extract structured parameters
        extraction_prompt = f"""
        Extract Telco customer churn prediction parameters from this query: "{query}"
        
        Return a JSON object with these fields (use null for missing values):
        {{
            "customer_id": "unique customer identifier",
            "age": "customer age",
            "tenure_months": "months as customer",
            "monthly_charges": "monthly charge amount",
            "total_charges": "total charges to date",
            "contract_type": "contract type (Month-to-month, One year, Two year)",
            "internet_service_type": "internet service (DSL, Fiber optic, No)",
            "tech_support": "has tech support (true/false)",
            "online_security": "has online security (true/false)",
            "support_tickets_count": "number of support tickets"
        }}
        
        Only return the JSON object, no other text.
        """
        
        params = self._call_gemini(extraction_prompt)
        logger.info(f"Extracted Churn parameters: {params}")
        return params
    
    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """
        Call Gemini API to process a prompt.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            Parsed response from Gemini
        """
        try:
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                'contents': [{
                    'parts': [{
                        'text': prompt
                    }]
                }]
            }
            
            url = f"{self.gemini_api_url}?key={self.api_key}"
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                
                # Try to parse as JSON
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse Gemini response as JSON: {text}")
                    return {}
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return {}
    
    def call_covid_service(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the COVID-19 prediction service.
        
        Args:
            parameters: Prediction parameters
            
        Returns:
            Prediction response from the service
        """
        logger.info(f"Calling COVID-19 service with parameters: {parameters}")
        
        try:
            response = requests.post(
                f"{self.covid_service_url}/predict/covid",
                json=parameters,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"COVID-19 service response: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calling COVID-19 service: {str(e)}")
            return {'error': str(e)}
    
    def call_churn_service(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the Telco Churn prediction service.
        
        Args:
            parameters: Prediction parameters
            
        Returns:
            Prediction response from the service
        """
        logger.info(f"Calling Churn service with parameters: {parameters}")
        
        try:
            response = requests.post(
                f"{self.churn_service_url}/predict/churn",
                json=parameters,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Churn service response: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calling Churn service: {str(e)}")
            return {'error': str(e)}
    
    def synthesize_response(self, intent: str, prediction: Dict[str, Any]) -> str:
        """
        Synthesize a natural language response from the model prediction.
        
        Args:
            intent: The classified intent ('covid' or 'churn')
            prediction: The prediction result from the model service
            
        Returns:
            Natural language response
        """
        logger.info(f"Synthesizing response for intent: {intent}")
        
        if 'error' in prediction:
            return f"I encountered an error while processing your request: {prediction['error']}"
        
        if intent == 'covid':
            return self._synthesize_covid_response(prediction)
        elif intent == 'churn':
            return self._synthesize_churn_response(prediction)
        else:
            return "I'm not sure how to interpret your question. Could you please clarify?"
    
    def _synthesize_covid_response(self, prediction: Dict[str, Any]) -> str:
        """Synthesize a response for COVID-19 predictions."""
        try:
            country = prediction.get('country_name', 'the country')
            risk_level = prediction.get('risk_level', 'UNKNOWN')
            confidence = prediction.get('confidence', 0)
            explanation = prediction.get('explanation', '')
            
            response = f"""
Based on the epidemiological data for {country}:

**Risk Assessment**: {risk_level}
**Confidence**: {confidence*100:.1f}%

{explanation}

**Recommendations**:
- Monitor case trends closely
- Ensure adequate vaccination coverage
- Maintain testing capacity
- Prepare healthcare infrastructure for potential surges
"""
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing COVID response: {str(e)}")
            return "I encountered an error while interpreting the prediction."
    
    def _synthesize_churn_response(self, prediction: Dict[str, Any]) -> str:
        """Synthesize a response for churn predictions."""
        try:
            customer_id = prediction.get('customer_id', 'the customer')
            churn_prob = prediction.get('churn_probability', 0)
            churn_pred = prediction.get('churn_prediction', False)
            risk_factors = prediction.get('risk_factors', [])
            retention_score = prediction.get('retention_score', 0)
            
            churn_status = "at risk of churning" if churn_pred else "likely to remain"
            
            response = f"""
**Customer {customer_id} Analysis**:

**Churn Risk**: {churn_prob*100:.1f}%
**Status**: This customer is {churn_status}.
**Retention Score**: {retention_score*100:.1f}%

**Key Risk Factors**:
"""
            for i, factor in enumerate(risk_factors, 1):
                response += f"\n{i}. {factor}"
            
            response += """

**Recommended Actions**:
- Reach out to the customer proactively
- Offer service improvements or discounts
- Address the identified pain points
- Monitor engagement metrics closely
"""
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing churn response: {str(e)}")
            return "I encountered an error while interpreting the prediction."
    
    def process_query(self, query: str) -> str:
        """
        Process a user query end-to-end.
        
        Args:
            query: User's natural language query
            
        Returns:
            Natural language response with prediction and recommendations
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Classify intent
            intent, confidence = self.classify_intent(query)
            
            if confidence < 0.3:
                return "I'm not sure if you're asking about COVID-19 or customer churn. Could you please clarify?"
            
            # Step 2: Extract parameters
            if intent == 'covid':
                parameters = self.extract_covid_parameters(query)
                if not parameters or 'country_name' not in parameters:
                    return "I need more information about the country to make a COVID-19 prediction. Please provide country name and relevant statistics."
                
                # Step 3: Call service
                prediction = self.call_covid_service(parameters)
            else:  # churn
                parameters = self.extract_churn_parameters(query)
                if not parameters or 'customer_id' not in parameters:
                    return "I need more information about the customer to assess churn risk. Please provide customer ID and relevant details."
                
                # Step 3: Call service
                prediction = self.call_churn_service(parameters)
            
            # Step 4: Synthesize response
            response = self.synthesize_response(intent, prediction)
            
            logger.info(f"Final response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"


def main():
    """Example usage of the Gemini Agent."""
    
    # Initialize agent
    api_key = os.getenv('GEMINI_API_KEY')
    covid_url = os.getenv('COVID_SERVICE_URL', 'http://localhost:8000')
    churn_url = os.getenv('CHURN_SERVICE_URL', 'http://localhost:8001')
    
    agent = GeminiAgent(api_key, covid_url, churn_url)
    
    # Example queries
    test_queries = [
        "What's the COVID-19 situation in the United States with 100000 cases and 2000 deaths?",
        "Is customer CUST_001 likely to churn? They've been with us for 24 months and pay $85.50/month.",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        response = agent.process_query(query)
        print(f"Response:\n{response}")


if __name__ == '__main__':
    main()
