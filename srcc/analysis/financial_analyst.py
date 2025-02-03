from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import base64
import os

class FinancialAnalystAI:
    def __init__(self, anthropic_api_key):
        self.llm = ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",
            temperature=0.4,
            max_tokens=1000,
            anthropic_api_key=anthropic_api_key
        )
        
        self.prompt_template = ChatPromptTemplate.from_template(
            """As a senior financial analyst, synthesize this market data and fundamental analysis:
            
            Market Conditions:
            {market_summary}
            
            Recent Patterns Detected:
            {patterns}
            
            Technical Indicators:
            - RSI: {rsi:.2f}
            - MACD: {macd_hist:.2f}
            - Bollinger Band Width: {bb_width:.2f}%
            
            Fundamental Metrics:
            - Revenue Growth: {revenue_growth:.1f}%
            - Gross Margin: {gross_margin:.1f}%
            - PE Ratio: {pe_ratio:.1f}
            - EPS Trend: {eps_trend}
            
            Generate a concise 4-paragraph analysis covering:
            1. Technical pattern confirmation and momentum
            2. Volatility and key levels
            3. Fundamental health assessment
            4. Integrated risk/reward evaluation""")
        
        self.vision_prompt = """Analyze this financial chart with the market data below:

Market Context:
{market_summary}

Technical Indicators:
- RSI: {rsi:.2f}
- MACD Histogram: {macd_hist:.2f}
- Bollinger Band Width: {bb_width:.2f}%

Provide analysis focusing on:
1. Chart pattern confirmation
2. Volume-price relationships
3. Key support/resistance levels visible
4. Probability assessment of continuation vs reversal"""
    
    def generate_report(self, market_summary, patterns, technicals):
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        return chain.run(
            market_summary=market_summary,
            patterns=patterns,
            **technicals
        )

    def analyze_chart_image(self, chart_path, market_data, technicals):
        # Add preprocessing validation
        if not os.path.exists(chart_path):
            return "Error: Chart image not found"
            
        if os.path.getsize(chart_path) > 5*1024*1024:  # 5MB limit
            return "Error: Chart image too large for analysis"
            
        # Create modified prompt with safe formatting
        vision_prompt = f"""{self.vision_prompt}
            
            Additional Technical Context:
            - Last Close: {technicals.get('close_price', 0):.2f}
            - Volume Trend: {"↑ Increasing" if technicals.get('volume', 0) > 0 else "↓ Decreasing"}
            - Pattern Intensity: {min(10, technicals.get('active_patterns_count', 0) * 2)}/10
            """
            
        # Encode image
        with open(chart_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": vision_prompt.format(
                        market_summary=market_data,
                        **technicals
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                }
            ]
        )
        
        return self.llm.invoke([message]).content 