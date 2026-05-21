import json
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any

from ai_waiter_core.agent.agent import get_agent_app
from ai_waiter_core.config import settings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Paths
E2E_DATA_FILES = [
    settings.PROJECT_ROOT / "evals/data/e2e/e2e_conversations_part1.json",
    settings.PROJECT_ROOT / "evals/data/e2e/e2e_conversations_part2.json"
]
RESULTS_DIR = settings.PROJECT_ROOT / "evals/results"
LOG_FILE = RESULTS_DIR / f"e2e_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
REPORT_FILE = RESULTS_DIR / "e2e_report.json"

os.makedirs(RESULTS_DIR, exist_ok=True)

def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_message + "\n")

def run_scenario(app, scenario: Dict[str, Any]) -> Dict[str, Any]:
    scenario_id = scenario['id']
    thread_id = f"eval_{scenario_id}_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    log(f"\n--- RUNNING SCENARIO {scenario_id}: {scenario['name']} ---")
    log(f"Description: {scenario['description']}")
    
    turns_results = []
    scenario_success = True
    
    for turn_data in scenario['turns']:
        turn_num = turn_data['turn']
        user_input = turn_data['content']
        expected_assertions = turn_data.get('assert', {})
        
        log(f"Turn {turn_num} [User]: {user_input}")
        
        # Invoke agent
        # table_id is expected in state
        input_state = {"messages": [HumanMessage(content=user_input)], "table_id": scenario['table_id']}
        output_state = app.invoke(input_state, config=config)
        
        # Inspect state for assertions
        messages = output_state['messages']
        # The last message is the AI response
        ai_response = messages[-1].content
        log(f"Turn {turn_num} [AI]: {ai_response}")
        
        # Find tool calls in this turn
        # Usually, if a tool was called, there will be an AIMessage with tool_calls 
        # followed by a ToolMessage
        turn_tool_calls = []
        turn_tool_outputs = []
        
        # We look at messages since the last HumanMessage
        # Actually, since we use LangGraph, we can just look at the new messages added in this turn
        # For simplicity, we search backwards until the HumanMessage we just sent
        new_messages = []
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content == user_input:
                break
            new_messages.append(msg)
        new_messages.reverse()
        
        for msg in new_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    turn_tool_calls.append(tc['name'])
            if isinstance(msg, ToolMessage):
                turn_tool_outputs.append(str(msg.content))
        
        # Assertions
        turn_success = True
        assertion_logs = []
        
        # 1. Tool Called
        expected_tool = expected_assertions.get('tool_called')
        if expected_tool:
            if expected_tool in turn_tool_calls:
                assertion_logs.append(f"  [PASS] Tool '{expected_tool}' called")
            else:
                assertion_logs.append(f"  [FAIL] Tool '{expected_tool}' NOT called. Actual: {turn_tool_calls}")
                turn_success = False
        
        # 2. Tool NOT Called
        not_expected_tool = expected_assertions.get('tool_must_NOT_call')
        if not_expected_tool:
            if not_expected_tool in turn_tool_calls:
                assertion_logs.append(f"  [FAIL] Tool '{not_expected_tool}' was called but shouldn't have been")
                turn_success = False
            else:
                assertion_logs.append(f"  [PASS] Tool '{not_expected_tool}' NOT called as expected")
        
        # 3. Tool Output Contains
        tool_output_check = expected_assertions.get('tool_output_contains')
        if tool_output_check:
            found = any(tool_output_check in out for out in turn_tool_outputs)
            if found:
                assertion_logs.append(f"  [PASS] Tool output contains '{tool_output_check}'")
            else:
                assertion_logs.append(f"  [FAIL] Tool output does NOT contain '{tool_output_check}'")
                turn_success = False
        
        # 4. Response Should Contain One Of
        response_one_of = expected_assertions.get('response_should_contain_one_of', [])
        if response_one_of:
            found = any(str(term).lower() in ai_response.lower() for term in response_one_of)
            if found:
                assertion_logs.append(f"  [PASS] Response contains one of expected terms")
            else:
                assertion_logs.append(f"  [FAIL] Response does NOT contain any of {response_one_of}")
                turn_success = False

        # 5. Response Contains
        response_contains = expected_assertions.get('response_contains')
        if response_contains:
            found_in_ai = response_contains.lower() in ai_response.lower()
            found_in_tool = any(response_contains.lower() in out.lower() for out in turn_tool_outputs)
            if found_in_ai or found_in_tool:
                assertion_logs.append(f"  [PASS] Response/Tool output contains '{response_contains}'")
            else:
                assertion_logs.append(f"  [FAIL] Response/Tool output does NOT contain '{response_contains}'")
                turn_success = False

        # 6. Confirmed Items Must Contain
        confirmed_must_contain = expected_assertions.get('confirmed_items_must_contain')
        if confirmed_must_contain:
            found_in_ai = confirmed_must_contain.lower() in ai_response.lower()
            found_in_tool = any(confirmed_must_contain.lower() in out.lower() for out in turn_tool_outputs)
            if found_in_ai or found_in_tool:
                assertion_logs.append(f"  [PASS] Confirmed items contain '{confirmed_must_contain}'")
            else:
                assertion_logs.append(f"  [FAIL] Confirmed items do NOT contain '{confirmed_must_contain}'")
                turn_success = False

        # 7. Confirmed Items Must NOT Contain
        confirmed_must_not_contain = expected_assertions.get('confirmed_items_must_NOT_contain')
        if confirmed_must_not_contain:
            found_in_ai = confirmed_must_not_contain.lower() in ai_response.lower()
            found_in_tool = any(confirmed_must_not_contain.lower() in out.lower() for out in turn_tool_outputs)
            if not found_in_ai and not found_in_tool:
                assertion_logs.append(f"  [PASS] Confirmed items do NOT contain '{confirmed_must_not_contain}' as expected")
            else:
                assertion_logs.append(f"  [FAIL] Confirmed items contain '{confirmed_must_not_contain}' but should not")
                turn_success = False
        
        for l in assertion_logs: log(l)
        
        turns_results.append({
            "turn": turn_num,
            "success": turn_success,
            "tool_calls": turn_tool_calls,
            "response": ai_response
        })
        
        if not turn_success:
            scenario_success = False
            
    return {
        "id": scenario_id,
        "name": scenario['name'],
        "success": scenario_success,
        "turns": turns_results
    }

def run_evaluation(limit: int = 4):
    log(f"Starting End-to-End (E2E) Evaluation (limit={limit if limit > 0 else 'All'})...")
    
    # Initialize Agent
    app = get_agent_app()
    
    all_scenario_results = []
    scenarios_to_run = []
    
    for data_file in E2E_DATA_FILES:
        if not os.path.exists(data_file):
            log(f"Warning: Data file {data_file} not found. Skipping.")
            continue
            
        with open(data_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        log(f"Loaded {len(dataset.get('scenarios', []))} scenarios from {os.path.basename(data_file)}")
        scenarios_to_run.extend(dataset.get('scenarios', []))
        
    if limit > 0:
        scenarios_to_run = scenarios_to_run[:limit]
        
    log(f"Executing {len(scenarios_to_run)} scenarios...")
    
    for scenario in scenarios_to_run:
        result = run_scenario(app, scenario)
        all_scenario_results.append(result)
            
    # Summary
    total = len(all_scenario_results)
    passed = sum(1 for r in all_scenario_results if r['success'])
    pass_rate = passed / total if total > 0 else 0
    
    log(f"\nE2E EVALUATION SUMMARY:")
    log(f"  Total Scenarios: {total}")
    log(f"  Passed:          {passed}")
    log(f"  Pass Rate:       {pass_rate:.2%}")
    
    # Save Report
    report = {
        "summary": {
            "timestamp": datetime.now().isoformat(),
            "pass_rate": pass_rate,
            "total_scenarios": total,
            "passed_count": passed
        },
        "results": all_scenario_results
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    log(f"\nFull E2E report saved to {REPORT_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run E2E evaluation for AI Waiter Agent.")
    parser.add_argument("--limit", type=int, default=4, help="Limit the number of scenarios to run (default: 4, set to -1 or 0 for all)")
    args = parser.parse_args()
    
    # If limit is 0 or negative, run all scenarios
    run_limit = args.limit if args.limit > 0 else -1
    run_evaluation(limit=run_limit)
