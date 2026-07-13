import asyncio
import json
import os
import re
import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from redis.asyncio import Redis

# Load Configurations
load_dotenv(dotenv_path="config/.env")
api_key = os.getenv("GEMINI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

class MoEOrchestrator:
    def __init__(self):
        self.redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.llm_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        # Using flash-latest to bypass billing tier restrictions while retaining deep reasoning
        self.model_name = "gemini-2.5-flash"
        print(f"\n[ORCHESTRATOR INIT] Using LLM Model: {self.model_name}")

    async def _query_llm(self, system_prompt, user_prompt):
        """Robust API caller with a Smart Circuit Breaker. Truncation Bug Eliminated."""
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # max_tokens REMOVED entirely to prevent API Wrapper truncation bugs
            "temperature": 0.4 # Encourages aggressive, creative debate
        }

        max_retries = 4  
        attempt = 0
        
        while attempt < max_retries:
            try:
                response = await self.llm_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
                
            except Exception as e:
                attempt += 1
                error_msg = str(e)
                print(f"   [API ERROR] Attempt {attempt}/{max_retries} failed: {error_msg}")
                if attempt == max_retries:
                    raise e
                    
                if "429" in error_msg:
                    match = re.search(r'retry in (\d+\.?\d*)s', error_msg)
                    if match:
                        wait_time = float(match.group(1)) + 1.0
                        print(f"   [RATE LIMIT] API Quota Exceeded. Sleeping for {wait_time:.1f} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        print("   [RATE LIMIT] API Quota Exceeded. Sleeping for 30 seconds...")
                        await asyncio.sleep(30)
                elif "Connection error" in error_msg or "ConnectError" in error_msg:
                    print(f"   [NETWORK] Connection dropped by Windows/ISP. Retrying in {5 * attempt} seconds...")
                    await asyncio.sleep(5 * attempt)
                else:
                    await asyncio.sleep(5 * attempt)

    async def _run_technical_agent(self, context_raw: str) -> str:
        """Agent 1: Generates the deep multi-timeframe synthesis."""
        sys_prompt = """
        You are the Lead Technical Analyst for a quantitative crypto fund.
        Read the provided JSON market state. 
        Write a HIGHLY DETAILED, 2-to-3 paragraph analytical thesis on the current Bitcoin price action.
        You MUST explicitly detail the relationship between the 1m/5m/15m micro-trends and the 1H/4H macro trends.
        Cite specific numbers (EMA distances, RSI, MACD, and KRS Zones).
        DO NOT cut your response short. Provide a comprehensive summary.
        """
        user_prompt = f"<market_data>\n{context_raw}\n</market_data>"
        
        print(" -> [AGENT] Technical Analyst synthesizing data...")
        return await self._query_llm(sys_prompt, user_prompt)

    async def _run_3_round_debate(self, sys_bull: str, sys_bear: str, base_context: str) -> tuple[str, str]:
        """Runs the deep 3-round adversarial loop between two competing logic vectors."""
        
        # ROUND 1: Opening Statements
        print(" -> [DEBATE] Round 1: Opening Statements...")
        bull_r1 = await self._query_llm(sys_bull, f"{base_context}\nWrite a detailed, 2-paragraph opening thesis supporting your position. Cite the data.")
        await asyncio.sleep(6) 
        bear_r1 = await self._query_llm(sys_bear, f"{base_context}\nWrite a detailed, 2-paragraph opening thesis supporting your position. Cite the data.")
        await asyncio.sleep(6)

        # ROUND 2: Rebuttals (Enforced Counter-Arguments)
        print(" -> [DEBATE] Round 2: Vicious Rebuttals...")
        bull_r2_prompt = f"{base_context}\n<opponent_argument>\n{bear_r1}\n</opponent_argument>\nYou MUST explicitly counter their argument. Quote their weak points and destroy their logic using the raw data. Write 2 full paragraphs."
        bear_r2_prompt = f"{base_context}\n<opponent_argument>\n{bull_r1}\n</opponent_argument>\nYou MUST explicitly counter their argument. Quote their weak points and destroy their logic using the raw data. Write 2 full paragraphs."
        
        bull_r2 = await self._query_llm(sys_bull, bull_r2_prompt)
        await asyncio.sleep(6)
        bear_r2 = await self._query_llm(sys_bear, bear_r2_prompt)
        await asyncio.sleep(6)

        # ROUND 3: Closing Predictions
        print(" -> [DEBATE] Round 3: Final Predictions...")
        bull_r3_prompt = f"{base_context}\n<opponent_rebuttal>\n{bear_r2}\n</opponent_rebuttal>\nWrite your absolute final, 1-paragraph conclusion on why your thesis will win despite the opponent's rebuttal."
        bear_r3_prompt = f"{base_context}\n<opponent_rebuttal>\n{bull_r2}\n</opponent_rebuttal>\nWrite your absolute final, 1-paragraph conclusion on why your thesis will win despite the opponent's rebuttal."
        
        bull_r3 = await self._query_llm(sys_bull, bull_r3_prompt)
        await asyncio.sleep(6)
        bear_r3 = await self._query_llm(sys_bear, bear_r3_prompt)
        
        # Compile transcripts
        bull_transcript = f"<round_1>\n{bull_r1}\n</round_1>\n<round_2>\n{bull_r2}\n</round_2>\n<round_3>\n{bull_r3}\n</round_3>"
        bear_transcript = f"<round_1>\n{bear_r1}\n</round_1>\n<round_2>\n{bear_r2}\n</round_2>\n<round_3>\n{bear_r3}\n</round_3>"
        
        return bull_transcript, bear_transcript

    def _extract_json_from_response(self, text: str) -> str:
        """Safely extracts the JSON object using aggressive hunting (bypasses Markdown)."""
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                possible_json = text[start_idx:end_idx+1]
                json.loads(possible_json)
                return possible_json
        except json.JSONDecodeError:
            pass
            
        print(f"\n[PARSE ERROR DEBUG] Raw LLM Output:\n{text}\n")
        return '{"direction": "NEUTRAL", "conviction_score": 0.0, "reasoning": "Parse Error: Failed to extract valid JSON.", "invalidation_condition": "None"}'

    # ==========================================
    # PIPELINE A: ALPHA HUNTER (ENTRY)
    # ==========================================
    async def _run_pipeline_a_entry(self, context_raw: str):
        print("\n[PIPELINE A] Alpha Hunter Active. State = FLAT. Searching for Entries...")
        ta_report = await self._run_technical_agent(context_raw)
        
        base_context = f"<technical_report>\n{ta_report}\n</technical_report>\n<raw_data>\n{context_raw}\n</raw_data>"
        
        bull_sys = "You are a permabull crypto quant. Use the data to build a savage, multi-paragraph mathematical case for going LONG. Do not hold back."
        bear_sys = "You are a permabear crypto quant. Use the data to build a savage, multi-paragraph mathematical case for going SHORT. Do not hold back."
        
        bull_tx, bear_tx = await self._run_3_round_debate(bull_sys, bear_sys, base_context)
        
        sys_prompt = """
        You are the Master Quantitative Portfolio Manager.
        Evaluate the raw data, the technical report, and the 3-round debate transcript.
        
        CRITICAL INSTRUCTIONS:
        1. First, you MUST output a <scratchpad> block where you explicitly reason through the arguments.
        2. Second, you MUST output a strict JSON block formatted exactly as requested.
        
        CRITICAL: The `invalidation_condition` MUST be a "Semantic Condition" (e.g., '1H CVD drops and 15m RSI < 30'), NOT a static price target. Static prices get hunted.
        
        JSON SCHEMA:
        {
          "direction": "LONG" or "SHORT" or "NEUTRAL",
          "conviction_score": <float between 0.0 and 1.0>,
          "reasoning": "<A highly detailed 3-4 sentence explanation. DO NOT cut off your sentence.>",
          "invalidation_condition": "<Semantic structural condition that proves this trade wrong>"
        }
        """
        
        user_prompt = f"{base_context}\n<bull_debate_transcript>\n{bull_tx}\n</bull_debate_transcript>\n<bear_debate_transcript>\n{bear_tx}\n</bear_debate_transcript>\n\nOutput your <scratchpad> reasoning, followed by the JSON execution block. Make sure your JSON is fully complete and closed with a right curly bracket."
        
        print(" -> [AGENT] Master AI evaluating debate and calculating execution vectors...")
        master_raw = await self._query_llm(sys_prompt, user_prompt)
        final_json_str = self._extract_json_from_response(master_raw)
        
        self._save_transcript("ENTRY", context_raw, ta_report, bull_tx, bear_tx, master_raw, final_json_str)
        return final_json_str

    # ==========================================
    # PIPELINE B: THESIS VALIDATOR (EXIT)
    # ==========================================
    async def _run_pipeline_b_exit(self, context_raw: str, active_trade_json: str):
        print("\n[PIPELINE B] Thesis Validator Active. State = IN_POSITION. Managing Open Position...")
        ta_report = await self._run_technical_agent(context_raw)
        
        base_context = f"<original_entry_thesis>\n{active_trade_json}\n</original_entry_thesis>\n<live_technical_report>\n{ta_report}\n</live_technical_report>\n<live_raw_data>\n{context_raw}\n</live_raw_data>"
        
        bull_sys = "You are the Position Manager. Our bot is currently holding an active trade. Your goal is to vigorously defend the original entry thesis, argue that current volatility is just noise, and push to HOLD."
        bear_sys = "You are the Risk Auditor. Our bot is holding a trade. Vigorously attack the original entry thesis. Use live data to prove the Semantic Invalidation Condition has been met, and push to ABORT."
        
        bull_tx, bear_tx = await self._run_3_round_debate(bull_sys, bear_sys, base_context)
        
        sys_prompt = """
        You are the Master Quantitative Risk Manager.
        Evaluate the live data and the debate against the ORIGINAL entry thesis. Has the original semantic invalidation condition been met? Is momentum exhausted?
        
        CRITICAL INSTRUCTIONS:
        1. First, you MUST output a <scratchpad> block where you explicitly reason through the arguments.
        2. Second, you MUST output a strict JSON block formatted exactly as requested.
        
        JSON SCHEMA:
        {
          "direction": "HOLD" or "REDUCE_EXPOSURE" or "ABORT",
          "conviction_score": <float between 0.0 and 1.0>,
          "reasoning": "<A highly detailed 3-4 sentence explanation comparing original thesis to live data.>",
          "invalidation_condition": "N/A"
        }
        """
        
        user_prompt = f"{base_context}\n<bull_debate_transcript>\n{bull_tx}\n</bull_debate_transcript>\n<bear_debate_transcript>\n{bear_tx}\n</bear_debate_transcript>\n\nOutput your <scratchpad> reasoning, followed by the JSON execution block. Make sure your JSON is fully complete and closed with a right curly bracket."
        
        print(" -> [AGENT] Master AI evaluating debate and computing exit strategy...")
        master_raw = await self._query_llm(sys_prompt, user_prompt)
        final_json_str = self._extract_json_from_response(master_raw)
        
        self._save_transcript("EXIT", context_raw, ta_report, bull_tx, bear_tx, master_raw, final_json_str)
        return final_json_str

    def _save_transcript(self, pipeline_type, context, ta, bull, bear, master, final_json):
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tx = f"===================================================\n"
        tx += f"🕒 [{pipeline_type}] DEBATE CYCLE: {timestamp}\n"
        tx += f"===================================================\n\n"
        tx += f"--- 📊 RAW MARKET CONTEXT ---\n{context}\n\n"
        tx += f"--- 🧑‍💻 TECHNICAL AGENT REPORT ---\n{ta}\n\n"
        tx += f"--- 🐂 BULL AGENT ---\n{bull}\n\n"
        tx += f"--- 🐻 BEAR AGENT ---\n{bear}\n\n"
        tx += f"--- 🧠 MASTER AI RAW OUTPUT (Incl. Scratchpad) ---\n{master}\n\n"
        tx += f"--- 🎯 FINAL JSON ---\n{final_json}\n===================================================\n"
        
        with open(os.path.join(log_dir, "full_debate_history.txt"), "a", encoding="utf-8") as f:
            f.write(tx)
        with open(os.path.join(log_dir, "latest_debate.txt"), "w", encoding="utf-8") as f:
            f.write(tx)

    def _log_training_data(self, context_raw: str, final_decision_json: str):
        log_dir = "data/training"
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "distillation_dataset.jsonl")
        
        training_pair = {
            "messages": [
                {"role": "system", "content": "You are an autonomous HFT agent. Output strict JSON."},
                {"role": "user", "content": f"Market State: {context_raw}"},
                {"role": "assistant", "content": final_decision_json}
            ]
        }
        
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_pair) + "\n")

    async def run_pipeline(self):
        print("===================================================")
        print("🧠 INITIATING STATE-AWARE AI ORCHESTRATOR")
        print("===================================================")
        
        while True:
            try:
                context_raw = await self.redis.get("ai_context:latest_1h_summary")
                if not context_raw:
                    await asyncio.sleep(5)
                    continue
                
                # FSM Check: Route logic based on open inventory
                active_trade = await self.redis.hget("btc_market_state:portfolio", "active_trade")
                
                if active_trade:
                    # STATE: IN_POSITION -> Execute Pipeline B
                    final_json_str = await self._run_pipeline_b_exit(context_raw, active_trade)
                else:
                    # STATE: FLAT -> Execute Pipeline A
                    final_json_str = await self._run_pipeline_a_entry(context_raw)
                
                try:
                    decision = json.loads(final_json_str)
                    print("\n===================================================")
                    print(f"🎯 FINAL DECISION: {decision.get('direction')} (Conviction: {decision.get('conviction_score')})")
                    print(f"🧠 REASONING: {decision.get('reasoning')}")
                    print(f"🛡️ INVALIDATION: {decision.get('invalidation_condition')}")
                    print("===================================================\n")
                    print("📁 Full transcript saved to: data/logs/latest_debate.txt")
                    
                    # Pass mandate to deterministic execution layer
                    await self.redis.hset("btc_market_state:ai_mandate", "latest_signal", final_json_str)
                    
                    # Log telemetry for future fine-tuning
                    self._log_training_data(context_raw, final_json_str)
                    
                except json.JSONDecodeError:
                     print("[ERROR] Master AI failed to output valid JSON. Check data/logs/latest_debate.txt")
                
                # Heavy sleep to prevent API burnout. Python Tier 1 Risk Engine handles fast microstructure!
                print("[ORCHESTRATOR] Cycle complete. Sleeping for 2 minutes...")
                await asyncio.sleep(120) 
                
            except Exception as e:
                print(f"\n[ORCHESTRATOR GLOBAL ERROR] Pipeline Crash: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    brain = MoEOrchestrator()
    try:
        asyncio.run(brain.run_pipeline())
    except KeyboardInterrupt:
        print("\n[SYSTEM] AI Brain offline.")